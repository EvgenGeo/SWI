# telegram_bot_FIXED.py
# Telegram бот для MASW проекта - ИСПРАВЛЕННАЯ ВЕРСИЯ
# Исправлена ошибка: "You cannot call this from an async context"
import tempfile
import os
import io
import pickle
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import django
from django.conf import settings

# Инициализация Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'masw.settings')
django.setup()

# ВАЖНО: Для синхронных вызовов БД в async контексте нужно использовать sync_to_async
from asgiref.sync import sync_to_async
from django.db import connections

from telegram import (
    Update, ReplyKeyboardMarkup, ReplyKeyboardRemove,
    InlineKeyboardMarkup, InlineKeyboardButton, InputFile
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    ConversationHandler, CallbackQueryHandler, ContextTypes
)
from telegram.constants import ChatAction

# Импорт моделей Django (обернем в sync_to_async)
from spectral_masw.models import SegyFile, SpectralResult, TraceHeader
from spectral_masw.processing import process_seismogram

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ==================== СОСТОЯНИЯ БОТА ====================
class BotState(Enum):
    MAIN_MENU = 0
    UPLOAD_SEGY = 1
    VIEW_SEGY_LIST = 2
    PROCESS_SEGY = 3
    VIEW_RESULTS_LIST = 4
    VIEW_RESULT = 5
    ENTER_PROCESS_PARAMS = 6


# ==================== SYNC_TO_ASYNC ОБЕРТКИ ====================

@sync_to_async
def get_all_segy_files():
    """Получить все SEGY-файлы из БД (async-safe)"""
    return list(SegyFile.objects.all().order_by('-id'))


@sync_to_async
def get_segy_by_id(segy_id: int):
    """Получить SEGY-файл по ID"""
    return SegyFile.objects.get(id=segy_id)


@sync_to_async
def get_all_results():
    """Получить все результаты анализа"""
    return list(SpectralResult.objects.all().order_by('-id'))


@sync_to_async
def get_result_by_id(result_id: int):
    """Получить результат по ID"""
    return SpectralResult.objects.get(id=result_id)


@sync_to_async
def delete_segy_file(segy_id: int):
    """Удалить SEGY-файл"""
    segy = SegyFile.objects.get(id=segy_id)
    segy.delete()


@sync_to_async
def delete_result(result_id: int):
    """Удалить результат анализа"""
    result = SpectralResult.objects.get(id=result_id)
    result.delete()


@sync_to_async
def save_segy_to_db_async(file_path: str, file_name: str) -> int:
    """Сохранить SEGY-файл в БД (async-safe)"""
    import segyio

    with segyio.open(file_path, ignore_geometry=True) as s:
        traces = np.array([s.trace[i][:] for i in range(s.tracecount)], dtype='float32')
        sample_rate = s.bin[segyio.BinField.Interval]

        segy_obj = SegyFile.objects.create(
            name=file_name,
            original_file=f'segy/{file_name}',
            sample_rate=sample_rate,
            num_samples=traces.shape[1],
            num_traces=traces.shape[0],
            trace_data=pickle.dumps(traces)
        )

        trace_headers_to_create = []
        for trace_idx in range(s.tracecount):
            h = s.header[trace_idx]
            trace_headers_to_create.append(
                TraceHeader(
                    segy_file=segy_obj,
                    trace_number=trace_idx,
                    ffig=int(h.get(segyio.TraceField.FieldRecord, 0)),
                    rec_x=float(h.get(segyio.TraceField.GroupX, 0)),
                    rec_y=float(h.get(segyio.TraceField.GroupY, 0)),
                    sou_x=float(h.get(segyio.TraceField.SourceX, 0)),
                    sou_y=float(h.get(segyio.TraceField.SourceY, 0)),
                    offset=float(h.get(segyio.TraceField.offset, 0)),
                    elevation=float(h.get(segyio.TraceField.ReceiverGroupElevation, 0)),
                    cdp_x=float(h.get(segyio.TraceField.CDP_X, 0)),
                    cdp_y=float(h.get(segyio.TraceField.CDP_Y, 0)),
                    cdp=int(h.get(segyio.TraceField.CDP, 0)),
                    dt=int(h.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL, s.bin[segyio.BinField.Interval]))
                )
            )

        TraceHeader.objects.bulk_create(trace_headers_to_create)
        return segy_obj.id


@sync_to_async
def get_trace_headers(segy_id: int):
    """Получить заголовки трасс"""
    return list(TraceHeader.objects.filter(segy_file_id=segy_id).order_by('trace_number'))


@sync_to_async
def create_spectral_result(segy_id: int, name: str, extent_x_min: float, extent_x_max: float,
                           extent_y_min: float, extent_y_max: float, processing_notes: str,
                           spectrum_data, curve_vel, curve_freq) -> int:
    """Создать результат спектрального анализа"""
    segy = SegyFile.objects.get(id=segy_id)

    spec_res = SpectralResult.objects.create(
        segy_file=segy,
        name=name,
        extent_x_min=extent_x_min,
        extent_x_max=extent_x_max,
        extent_y_min=extent_y_min,
        extent_y_max=extent_y_max,
        processing_notes=processing_notes
    )

    spec_res.set_spectrogram(spectrum_data)
    spec_res.set_curve_arrays(curve_vel, curve_freq)
    spec_res.save()

    return spec_res.id

# ==================== ФУНКЦИИ ОБРАБОТКИ ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Стартовая команда - показываем главное меню"""
    user = update.effective_user
    welcome_text = (
        f"🌍 Добро пожаловать, {user.first_name}!\n\n"
        "Это бот для обработки сейсмических данных MASW.\n"
        "Выберите действие:"
    )

    keyboard = [
        ["📊 Загрузить SEGY-файл"],
        ["📈 Список сейсмограмм"],
        ["🎯 Результаты анализа"],
        ["❌ Выход"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

    await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    return BotState.MAIN_MENU.value


# async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
#     """Обработка выбора в главном меню"""
#     text = update.message.text
#
#     if text == "📊 Загрузить SEGY-файл":
#         msg = (
#             "📁 Отправьте SEGY-файл для загрузки.\n\n"
#             "Поддерживаемые форматы: .segy, .sgy"
#         )
#         await update.message.reply_text(msg)
#         return BotState.UPLOAD_SEGY.value
#
#     elif text == "📈 Список сейсмограмм":
#         await show_segy_list(update, context)
#         return BotState.VIEW_SEGY_LIST.value
#
#     elif text == "🎯 Результаты анализа":
#         await show_results_list(update, context)
#         return BotState.VIEW_RESULTS_LIST.value
#
#     elif text == "❌ Выход":
#         await update.message.reply_text(
#             "До свидания! 👋",
#             reply_markup=ReplyKeyboardRemove()
#         )
#         return ConversationHandler.END
#
#     else:
#         await update.message.reply_text("❌ Неизвестная команда. Выберите из меню.")
#         return BotState.MAIN_MENU.value

async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text

    if text in ["🏠 Главное меню", "/start"]:
        await update.message.reply_text("Выберите действие из меню ниже.")
        return BotState.MAIN_MENU.value

    if text == "📊 Загрузить SEGY-файл":
        msg = (
            "📁 Отправьте SEGY-файл для загрузки.\n\n"
            "Поддерживаемые форматы: .segy, .sgy"
        )
        await update.message.reply_text(msg)
        return BotState.UPLOAD_SEGY.value

    elif text == "📈 Список сейсмограмм":
        await show_segy_list(update, context)
        return BotState.VIEW_SEGY_LIST.value

    elif text == "🎯 Результаты анализа":
        await show_results_list(update, context)
        return BotState.VIEW_RESULTS_LIST.value

    elif text == "❌ Выход":
        await update.message.reply_text(
            "До свидания! 👋",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    else:
        await update.message.reply_text("❌ Неизвестная команда. Выберите из меню.")
        return BotState.MAIN_MENU.value


async def upload_segy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение и сохранение SEGY-файла"""
    if not update.message.document:
        await update.message.reply_text("❌ Пожалуйста, отправьте файл.")
        return BotState.UPLOAD_SEGY.value

    try:
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)

        file = update.message.document
        file_name = file.file_name

        # Проверка расширения
        if not file_name.lower().endswith(('.segy', '.sgy')):
            await update.message.reply_text(
                "❌ Поддерживаются только файлы .segy или .sgy"
            )
            return BotState.UPLOAD_SEGY.value

        # Скачиваем файл
        tg_file = await context.bot.get_file(file.file_id)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)
        await tg_file.download_to_drive(temp_path)

        # Сохраняем в БД через async wrapper
        await save_segy_to_db_async(temp_path, file_name)

        # Очищаем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)

        await update.message.reply_text(
            f"✅ Файл '{file_name}' успешно загружен!\n\n"
            "Вернитесь в главное меню для дальнейших действий.",
            reply_markup=get_main_menu_keyboard()
        )
        return BotState.MAIN_MENU.value

    except Exception as e:
        logger.error(f"Ошибка при загрузке SEGY: {e}")
        await update.message.reply_text(f"❌ Ошибка при загрузке: {str(e)}")
        return BotState.UPLOAD_SEGY.value


async def show_segy_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Показываем список загруженных SEGY-файлов"""
    try:
        segy_files = await get_all_segy_files()

        if not segy_files:
            await update.message.reply_text(
                "📭 Нет загруженных сейсмограмм.\n\n"
                "Загрузите первый файл через меню '📊 Загрузить SEGY-файл'.",
                reply_markup=get_main_menu_keyboard()
            )
            return BotState.MAIN_MENU.value

        # Создаем инлайн кнопки для каждого файла
        buttons = []
        for segy in segy_files:
            btn_text = f"📄 {segy.name[:30]} ({segy.num_traces} трасс)"
            buttons.append([
                InlineKeyboardButton(
                    btn_text,
                    callback_data=f"select_segy_{segy.id}"
                )
            ])

        buttons.append([InlineKeyboardButton("« Назад", callback_data="back_to_menu")])
        reply_markup = InlineKeyboardMarkup(buttons)

        text = f"📊 Доступно сейсмограмм: {len(segy_files)}\n\nВыберите файл:"
        await update.message.reply_text(text, reply_markup=reply_markup)

        return BotState.VIEW_SEGY_LIST.value

    except Exception as e:
        logger.error(f"Ошибка при получении списка SEGY: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def select_segy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора SEGY-файла"""
    query = update.callback_query
    await query.answer()

    try:
        segy_id = int(query.data.split('_')[2])
        segy = await get_segy_by_id(segy_id)

        context.user_data['selected_segy_id'] = segy_id

        info_text = (
            f"📊 Информация о сейсмограмме:\n\n"
            f"<b>Название:</b> {segy.name}\n"
            f"<b>Трасс:</b> {segy.num_traces}\n"
            f"<b>Отсчетов:</b> {segy.num_samples}\n"
            f"<b>Частота дискретизации:</b> {segy.sample_rate} мкс\n\n"
            f"<b>Выберите действие:</b>"
        )

        buttons = [
            [InlineKeyboardButton("🔧 Обработать", callback_data="start_process")],
            [InlineKeyboardButton("🗑️ Удалить", callback_data="delete_segy")],
            [InlineKeyboardButton("« Назад", callback_data="back_to_segy_list")]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)

        await query.edit_message_text(info_text, reply_markup=reply_markup, parse_mode='HTML')
        return BotState.VIEW_SEGY_LIST.value

    except Exception as e:
        logger.error(f"Ошибка при выборе SEGY: {e}")
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def start_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Начало процесса обработки - запрос параметров"""
    query = update.callback_query
    await query.answer()

    context.user_data['params'] = {}

    text = (
        "🔧 Введите параметры обработки:\n\n"
        "1️⃣  <b>Минимальная частота (f_min)</b>, Гц\n"
        "Пример: 5"
    )

    await query.edit_message_text(text, parse_mode='HTML')
    return BotState.ENTER_PROCESS_PARAMS.value


async def enter_process_params(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получение параметров обработки"""
    try:
        # Определяем, какой параметр вводим
        params_count = len(context.user_data.get('params', {}))

        if params_count == 0:
            # F_MIN
            value = float(update.message.text.strip())
            if value <= 0:
                await update.message.reply_text("❌ f_min должна быть > 0")
                return BotState.ENTER_PROCESS_PARAMS.value
            context.user_data['params']['f_min'] = value
            next_param = "2️⃣  <b>Максимальная частота (f_max)</b>, Гц\nПример: 50"

        elif params_count == 1:
            # F_MAX
            value = float(update.message.text.strip())
            f_min = context.user_data['params']['f_min']
            if value <= f_min:
                await update.message.reply_text("❌ f_max должна быть > f_min")
                return BotState.ENTER_PROCESS_PARAMS.value
            context.user_data['params']['f_max'] = value
            next_param = "3️⃣  <b>Минимальная скорость (v_min)</b>, м/с\nПример: 100"

        elif params_count == 2:
            # V_MIN
            value = float(update.message.text.strip())
            if value <= 0:
                await update.message.reply_text("❌ v_min должна быть > 0")
                return BotState.ENTER_PROCESS_PARAMS.value
            context.user_data['params']['v_min'] = value
            next_param = "4️⃣  <b>Максимальная скорость (v_max)</b>, м/с\nПример: 500"

        elif params_count == 3:
            # V_MAX
            value = float(update.message.text.strip())
            v_min = context.user_data['params']['v_min']
            if value <= v_min:
                await update.message.reply_text("❌ v_max должна быть > v_min")
                return BotState.ENTER_PROCESS_PARAMS.value
            context.user_data['params']['v_max'] = value
            next_param = "5️⃣  <b>Тип данных</b>\n✅ Введите: <b>2d</b> или <b>3d</b>"

        elif params_count == 4:
            # DATA_TYPE - ВАЛИДАЦИЯ!
            data_type = update.message.text.strip().lower()

            # Проверка: только 2d или 3d
            if data_type not in ['2d', '3d']:
                await update.message.reply_text(
                    "❌ Ошибка! Тип данных может быть только <b>2d</b> или <b>3d</b>\n\n"
                    "5️⃣  Пожалуйста, введите еще раз: <b>2d</b> или <b>3d</b>",
                    parse_mode='HTML'
                )
                return BotState.ENTER_PROCESS_PARAMS.value

            context.user_data['params']['data_type'] = data_type

            # Все параметры собраны - начинаем обработку
            await update.message.reply_text(
                "⏳ Обработка данных... Это может занять некоторое время.",
                reply_markup=ReplyKeyboardRemove()
            )

            await process_seismogram_async(update, context)
            return BotState.MAIN_MENU.value

        text = f"✅ Параметр сохранен.\n\n{next_param}"
        await update.message.reply_text(text, parse_mode='HTML')
        return BotState.ENTER_PROCESS_PARAMS.value

    except ValueError as e:
        params_count = len(context.user_data.get('params', {}))
        if params_count < 4:
            await update.message.reply_text(
                "❌ Ошибка формата. Пожалуйста, введите <b>число</b>.",
                parse_mode='HTML'
            )
        return BotState.ENTER_PROCESS_PARAMS.value

    except Exception as e:
        logger.error(f"Ошибка при вводе параметров: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def process_seismogram_async(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Асинхронная обработка сейсмограммы"""
    try:
        await update.message.chat.send_action(ChatAction.TYPING)

        segy_id = context.user_data.get('selected_segy_id')
        segy = await get_segy_by_id(segy_id)
        params = context.user_data.get('params', {})

        # Подготовка данных
        traces = pickle.loads(segy.trace_data)
        traces = traces.astype('float32')
        headers_qs = await get_trace_headers(segy_id)

        # Убедиться, что все данные float
        headers = np.array([
            [float(h.trace_number), float(h.rec_x), float(h.rec_y),
             float(h.sou_x), float(h.sou_y), float(h.elevation),
             float(h.cdp_x), float(h.offset), float(h.cdp_y), float(h.cdp)]
            for h in headers_qs
        ], dtype='float32').T

        # Преобразуем в float перед передачей
        f_min = float(params.get('f_min', 5))
        f_max = float(params.get('f_max', 50))
        v_min = float(params.get('v_min', 100))
        v_max = float(params.get('v_max', 1000))
        data_type = str(params.get('data_type', '2d')).lower()
        dt = float(segy.sample_rate)  # ГЛАВНОЕ: dt должен быть float!

        logger.info(f"Запуск обработки: traces={traces.shape}, headers={headers.shape}, dt={dt}")

        # Запуск обработки (может быть долгой - в отдельном потоке)
        result = await asyncio.to_thread(
            process_seismogram,
            traces=traces,
            headers=headers,
            dt=dt,
            data_type=data_type,
            f_min=f_min,
            f_max=f_max,
            v_min=v_min,
            v_max=v_max
        )

        logger.info(f"Обработка завершена: spectrum={result['spec_image'].shape}")

        # Сохранение результата в БД
        result_id = await create_spectral_result(
            segy_id=segy_id,
            name=f"Анализ {segy.name}",
            extent_x_min=float(result['vel_axis'].min()),
            extent_x_max=float(result['vel_axis'].max()),
            extent_y_min=float(result['freq_axis'].min()),
            extent_y_max=float(result['freq_axis'].max()),
            processing_notes=str(params),
            spectrum_data=result['spec_image'],
            curve_vel=result['curve_vel'],
            curve_freq=result['curve_freq']
        )

        # Отправка результата
        text = (
            f"✅ Обработка завершена!\n\n"
            f"<b>Параметры:</b>\n"
            f"f: {f_min}-{f_max} Гц\n"
            f"v: {v_min}-{v_max} м/с\n\n"
            f"Результат сохранен в базе данных."
        )

        keyboard = [
            ["🎯 Результаты анализа"],
            ["📈 Список сейсмограмм"],
            ["🏠 Главное меню"]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode='HTML')

    except Exception as e:
        logger.error(f"Ошибка при обработке: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Ошибка при обработке: {str(e)}")

async def show_results_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Показываем список результатов анализа"""
    try:
        results = await get_all_results()

        if not results:
            await update.message.reply_text(
                "📭 Нет результатов анализа.\n\n"
                "Сначала обработайте сейсмограмму.",
                reply_markup=get_main_menu_keyboard()
            )
            return BotState.MAIN_MENU.value

        buttons = []
        for result in results:
            # ✅ Обертка для доступа к result.segy_file.name
            segy_name = await sync_to_async(lambda r=result: r.segy_file.name)()
            btn_text = f"📊 {result.name[:30]} ({segy_name[:20]})"
            buttons.append([
                InlineKeyboardButton(
                    btn_text,
                    callback_data=f"view_result_{result.id}"
                )
            ])

        buttons.append([InlineKeyboardButton("« Назад", callback_data="back_to_menu")])
        reply_markup = InlineKeyboardMarkup(buttons)

        text = f"🎯 Доступно результатов: {len(results)}\n\nВыберите для просмотра:"
        await update.message.reply_text(text, reply_markup=reply_markup)

        return BotState.VIEW_RESULTS_LIST.value

    except Exception as e:
        logger.error(f"Ошибка при получении списка результатов: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def view_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Просмотр результата с графиком"""
    query = update.callback_query
    await query.answer()

    try:
        await query.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

        result_id = int(query.data.split('_')[2])
        result = await get_result_by_id(result_id)

        # ✅ Обертки для синхронных методов БД
        spec = await sync_to_async(result.get_spectrogram)()
        v_min, v_max = result.extent_x_min, result.extent_x_max
        f_min, f_max = result.extent_y_min, result.extent_y_max

        curve_v = await sync_to_async(result.get_curve_x)()
        curve_f = await sync_to_async(result.get_curve_y)()

        # ✅ ПРОВЕРКА: Спектр не пустой?
        if spec is None or spec.size == 0:
            await query.edit_message_text("❌ Данные спектра повреждены или пусты!")
            return BotState.MAIN_MENU.value

        # ✅ ПРОВЕРКА: Кривые не пусты?
        if curve_v is None or curve_f is None or len(curve_v) == 0 or len(curve_f) == 0:
            await query.edit_message_text("❌ Данные кривой повреждены или пусты!")
            return BotState.MAIN_MENU.value

        logger.info(f"spec shape: {spec.shape}, curve_v shape: {curve_v.shape if hasattr(curve_v, 'shape') else len(curve_v)}")

        # Создание графика (в отдельном потоке)
        try:
            buf = await asyncio.to_thread(
                create_spectrum_plot,
                spec=spec,
                f_min=f_min, f_max=f_max,
                v_min=v_min, v_max=v_max,
                curve_v=curve_v, curve_f=curve_f,
                title=result.name
            )
        except Exception as plot_error:
            logger.error(f"Ошибка при создании графика: {plot_error}", exc_info=True)
            await query.edit_message_text(f"❌ Ошибка при создании графика: {str(plot_error)}")
            return BotState.MAIN_MENU.value

        # ✅ ПРОВЕРКА: Буфер не пустой?
        if buf.getbuffer().nbytes == 0:
            await query.edit_message_text("❌ График не создан (пустой буфер)!")
            return BotState.MAIN_MENU.value

        # Информация о результате
        info_text = (
            f"<b>📊 Результат спектрального анализа</b>\n\n"
            f"<b>Имя:</b> {result.name}\n"
            f"<b>Файл:</b> {await sync_to_async(lambda: result.segy_file.name)()}\n"
            f"<b>Частоты:</b> {f_min:.1f}-{f_max:.1f} Гц\n"
            f"<b>Скорости:</b> {v_min:.1f}-{v_max:.1f} м/с\n"
            f"<b>Параметры:</b> {result.processing_notes}\n"
        )

        buttons = [
            [InlineKeyboardButton("🗑️ Удалить", callback_data=f"delete_result_{result_id}")],
            [InlineKeyboardButton("« Назад", callback_data="back_to_results")]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)

        # Отправка изображения
        input_file = InputFile(buf, filename=f"result_{result_id}.png")
        await query.message.reply_photo(
            input_file,
            caption=info_text,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        return BotState.VIEW_RESULT.value

    except Exception as e:
        logger.error(f"Ошибка при просмотре результата: {e}", exc_info=True)
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


def create_spectrum_plot(spec, f_min, f_max, v_min, v_max, curve_v, curve_f, title):
    """Создание графика спектра (синхронная функция для asyncio.to_thread)"""
    try:
        # ✅ Валидация входных данных
        if spec is None or spec.size == 0:
            raise ValueError("Спектр пустой или None")

        if curve_v is None or curve_f is None:
            raise ValueError("Кривые пустые или None")

        # Преобразуем в numpy если нужно
        spec = np.array(spec) if not isinstance(spec, np.ndarray) else spec
        curve_v = np.array(curve_v) if not isinstance(curve_v, np.ndarray) else curve_v
        curve_f = np.array(curve_f) if not isinstance(curve_f, np.ndarray) else curve_f

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        im = ax.imshow(
            spec,
            origin='lower',
            aspect='auto',
            extent=[f_min, f_max, v_min, v_max],
            cmap='viridis'
        )

        # ✅ Проверка размеров кривых перед печатью
        if len(curve_v) > 0 and len(curve_f) > 0:
            ax.plot(curve_v, curve_f, 'r-', linewidth=2, label='Дисперсионная кривая')
            ax.legend()

        ax.set_ylabel('Скорость (м/с)')
        ax.set_xlabel('Частота (Гц)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Амплитуда')

        # Сохранение в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close(fig)

        return buf

    except Exception as e:
        logger.error(f"Ошибка при создании графика: {e}", exc_info=True)
        raise


async def delete_result_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Удаление результата анализа"""
    query = update.callback_query
    await query.answer()

    try:
        result_id = int(query.data.split('_')[2])
        await delete_result(result_id)  # ✅ Вызывает функцию-обертку из начала кода

        await query.edit_message_text(
            "✅ Результат удален.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("« Назад", callback_data="back_to_results")
            ]])
        )

        return BotState.MAIN_MENU.value

    except Exception as e:
        logger.error(f"Ошибка при удалении результата: {e}")
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def delete_segy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Удаление SEGY-файла"""
    query = update.callback_query
    await query.answer()

    try:
        segy_id = context.user_data.get('selected_segy_id')
        segy = await get_segy_by_id(segy_id)
        segy_name = segy.name

        await delete_segy_file(segy_id)

        await query.edit_message_text(
            f"✅ Файл '{segy_name}' удален.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("« Назад", callback_data="back_to_segy_list")
            ]])
        )

        return BotState.MAIN_MENU.value

    except Exception as e:
        logger.error(f"Ошибка при удалении SEGY: {e}")
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")
        return BotState.MAIN_MENU.value


async def back_to_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Возврат в главное меню"""
    query = update.callback_query
    await query.answer()

    user = query.from_user
    welcome_text = (
        f"🌍 Главное меню\n\n"
        f"Добро пожаловать, {user.first_name}!"
    )

    buttons = [
        [InlineKeyboardButton("📊 Загрузить SEGY", callback_data="upload")],
        [InlineKeyboardButton("📈 Список сейсмограмм", callback_data="view_segy_list")],
        [InlineKeyboardButton("🎯 Результаты анализа", callback_data="view_results")],
    ]
    reply_markup = InlineKeyboardMarkup(buttons)

    await query.edit_message_text(welcome_text, reply_markup=reply_markup)
    return BotState.MAIN_MENU.value


async def back_to_segy_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Возврат к списку SEGY"""
    query = update.callback_query
    await query.answer()

    segy_files = await get_all_segy_files()

    buttons = []
    for segy in segy_files:
        btn_text = f"📄 {segy.name[:30]} ({segy.num_traces} трасс)"
        buttons.append([
            InlineKeyboardButton(
                btn_text,
                callback_data=f"select_segy_{segy.id}"
            )
        ])

    buttons.append([InlineKeyboardButton("« Назад", callback_data="back_to_menu")])
    reply_markup = InlineKeyboardMarkup(buttons)

    text = f"📊 Доступно сейсмограмм: {len(segy_files)}\n\nВыберите файл:"
    await query.edit_message_text(text, reply_markup=reply_markup)

    return BotState.VIEW_SEGY_LIST.value


async def back_to_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Возврат к списку результатов"""
    query = update.callback_query
    await query.answer()

    try:
        results = await get_all_results()
        buttons = []

        for result in results:
            segy_name = await sync_to_async(lambda r=result: r.segy_file.name)()
            btn_text = f"📊 {result.name[:30]} ({segy_name[:20]})"
            buttons.append([
                InlineKeyboardButton(
                    btn_text,
                    callback_data=f"view_result_{result.id}"
                )
            ])

        buttons.append([InlineKeyboardButton("« Назад", callback_data="back_to_menu")])
        reply_markup = InlineKeyboardMarkup(buttons)
        text = f"🎯 Доступно результатов: {len(results)}\n\nВыберите для просмотра:"

        # ✅ ИСПРАВЛЕНИЕ: Проверяем тип сообщения
        message = query.message
        if message.photo:  # Если это фото (после просмотра результата)
            # Удаляем фото и отправляем новое текстовое сообщение
            await message.delete()
            await query.message.reply_text(text, reply_markup=reply_markup)
        else:  # Обычное текстовое сообщение
            await query.edit_message_text(text, reply_markup=reply_markup)

        return BotState.VIEW_RESULTS_LIST.value

    except Exception as e:
        logger.error(f"Ошибка при возврате к результатам: {e}")
        # ✅ ИСПРАВЛЕНИЕ: Всегда используем answer() + новое сообщение при ошибке
        await query.answer("Ошибка при загрузке списка", show_alert=True)
        return BotState.MAIN_MENU.value


def get_main_menu_keyboard():
    """Возвращает клавиатуру главного меню"""
    keyboard = [
        ["📊 Загрузить SEGY-файл"],
        ["📈 Список сейсмограмм"],
        ["🎯 Результаты анализа"],
        ["❌ Выход"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


def main():
    """Запуск бота"""
    TOKEN = "token"

    if TOKEN == 'YOUR_TOKEN_HERE':
        print("❌ Установите TELEGRAM_BOT_TOKEN в переменные окружения!")
        return

    # Создание приложения
    application = Application.builder().token(TOKEN).build()

    # Обработчик разговора
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            BotState.MAIN_MENU.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu)
            ],
            BotState.UPLOAD_SEGY.value: [
                MessageHandler(filters.Document.ALL, upload_segy),
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: None)
            ],
            BotState.VIEW_SEGY_LIST.value: [
                CallbackQueryHandler(select_segy, pattern=r"^select_segy_\d+$"),
                CallbackQueryHandler(back_to_menu, pattern="^back_to_menu$"),
                CallbackQueryHandler(back_to_segy_list, pattern="^back_to_segy_list$"),
                CallbackQueryHandler(start_process, pattern="^start_process$"),
                CallbackQueryHandler(delete_segy, pattern="^delete_segy$"),
            ],
            BotState.ENTER_PROCESS_PARAMS.value: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, enter_process_params)
            ],
            BotState.VIEW_RESULTS_LIST.value: [
                CallbackQueryHandler(view_result, pattern=r"^view_result_\d+$"),
                CallbackQueryHandler(back_to_menu, pattern="^back_to_menu$"),
                CallbackQueryHandler(back_to_results, pattern="^back_to_results$"),
            ],
            BotState.VIEW_RESULT.value: [
                CallbackQueryHandler(delete_result_handler, pattern=r"^delete_result_\d+$"),
                CallbackQueryHandler(back_to_results, pattern="^back_to_results$"),
            ],
        },
        fallbacks=[
            CommandHandler('start', start),
            CallbackQueryHandler(back_to_menu, pattern="^back_to_menu$"),
            CallbackQueryHandler(back_to_segy_list, pattern="^back_to_segy_list$"),
            CallbackQueryHandler(back_to_results, pattern="^back_to_results$"),
        ]
    )

    application.add_handler(conv_handler)

    # Запуск
    print("🚀 Бот запущен!")
    application.run_polling()


if __name__ == '__main__':

    main()
