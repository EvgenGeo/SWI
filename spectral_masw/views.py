# spectral_masw/views.py
import pickle
import numpy as np
import io, base64
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.core.files.storage import default_storage
import segyio

from .models import SegyFile, TraceHeader, SpectralResult
from .processing import process_seismogram



def index_view(request):
    # вариант 1: сразу редирект на список сейсмограмм
    return render(request, 'spectral_masw/index.html')

# @login_required
def segy_list_view(request):
    """Вкладка 1: список сейсмограмм"""
    segy_files = SegyFile.objects.all().order_by('id')
    return render(request, 'spectral_masw/segy_list.html', {
        'segy_files': segy_files,
    })

# @require_POST
def segy_delete_view(request, pk):
    segy = get_object_or_404(SegyFile, pk=pk)
    segy.delete()  # каскадно удалит связанные TraceHeader и SpectralResult
    return redirect('spectral_masw:segy_list')

# @login_required
@require_http_methods(["POST"])
def segy_upload_view(request):
    """Шаг 1: загрузка одного или нескольких SEGY-файлов в БД."""
    files = request.FILES.getlist('segy_files')
    created_ids = []

    for f in files:
        path = default_storage.save(f'segy/{f.name}', f)
        full_path = default_storage.path(path)

        with segyio.open(full_path, ignore_geometry=True) as s:
            traces = np.array([s.trace[i][:] for i in range(s.tracecount)], dtype='float32')
            sample_rate = s.bin[segyio.BinField.Interval]  # подстрой под свой формат

            segy_obj = SegyFile.objects.create(
                name=f.name,
                original_file=path,
                sample_rate=sample_rate,
                num_samples=traces.shape[1],
                num_traces=traces.shape[0],
                trace_data=pickle.dumps(traces),
            )

            trace_headers_to_create = []

            for trace_idx in range(s.tracecount):
                h = s.header[trace_idx]  # Заголовок трассы

                # Извлекаем нужные поля (подстрой под свой формат SEGY)
                trace_headers_to_create.append(
                    TraceHeader(
                        segy_file=segy_obj,
                        trace_number=trace_idx,
                        ffig=int(h.get(segyio.TraceField.FieldRecord, 0)),

                        # Координаты приёмника
                        rec_x=float(h.get(segyio.TraceField.GroupX, 0)),
                        rec_y=float(h.get(segyio.TraceField.GroupY, 0)),

                        # Координаты источника
                        sou_x=float(h.get(segyio.TraceField.SourceX, 0)),
                        sou_y=float(h.get(segyio.TraceField.SourceY, 0)),

                        # Расстояние и высота
                        offset=float(h.get(segyio.TraceField.offset, 0)),
                        elevation=float(h.get(segyio.TraceField.ReceiverGroupElevation, 0)),

                        # CDP
                        cdp_x=float(h.get(segyio.TraceField.CDP_X, 0)),
                        cdp_y=float(h.get(segyio.TraceField.CDP_Y, 0)),
                        cdp=int(h.get(segyio.TraceField.CDP, 0)),

                        # Дискретизация
                        dt=int(h.get(segyio.TraceField.TRACE_SAMPLE_INTERVAL,
                                     s.bin[segyio.BinField.Interval])),
                    )
                )

            # Bulk create для быстрого сохранения
            TraceHeader.objects.bulk_create(trace_headers_to_create)

            created_ids.append(segy_obj.id)
            print(f"✅ Файл {f.name}: загружено {s.tracecount} трасс и их заголовков")

    return JsonResponse({'ok': True, 'created': created_ids})

# @require_POST
def spectra_delete_view(request, pk):
    """Удаление результата спектрального анализа пользователем."""
    res = get_object_or_404(SpectralResult, pk=pk)
    res.delete()
    return redirect('spectral_masw:spectra_list')

# @login_required
def segy_process_view(request, pk):
    """
    Шаг 2–7:
    - GET: форма ввода (f_min, f_max, v_min, v_max).
    - POST: вызывает твой спектральный анализ и сохраняет SpectralResult.
    """
    segy = get_object_or_404(SegyFile, pk=pk)

    if request.method == 'GET':
        return render(request, 'spectral_masw/segy_process.html', {
            'segy': segy,
        })

    # POST — запускаем обработку
    f_min = float(request.POST['f_min'])
    f_max = float(request.POST['f_max'])
    v_min = float(request.POST['v_min'])
    v_max = float(request.POST['v_max'])
    data_type = str(request.POST['data_type'])
    # model_name = request.POST.get('model_name', 'default_model')

    traces = pickle.loads(segy.trace_data)  # (n_traces x n_samples)

    headers_qs = TraceHeader.objects.filter(segy_file=segy).order_by('trace_number')
    headers = np.array([
        [
            h.trace_number,
            h.rec_x,
            h.rec_y,
            h.sou_x,
            h.sou_y,
            h.elevation,
            h.cdp_x,
            h.offset,
            h.cdp_y,
            h.cdp,
        ]
        for h in headers_qs
    ], dtype='float32').T

    print(traces.shape, headers.shape, data_type)

    result = process_seismogram(
        traces=traces,
        headers=np.array(headers),
        dt=segy.sample_rate,
        data_type=data_type,
        f_min=f_min,
        f_max=f_max,
        v_min=v_min,
        v_max=v_max,
    )

    spec = result['spec_image']
    freq_axis = result['freq_axis']
    vel_axis = result['vel_axis']
    curve_f = result['curve_freq']
    curve_v = result['curve_vel']

    spec_res = SpectralResult.objects.create(
        segy_file=segy,
        name=f'Спектральный анализ {segy.name}',
        extent_x_min=float(vel_axis.min()),
        extent_x_max=float(vel_axis.max()),
        extent_y_min=float(freq_axis.min()),
        extent_y_max=float(freq_axis.max()),
        processing_notes=f'f=[{f_min},{f_max}], v=[{v_min},{v_max}]',
    )
    spec_res.set_spectrogram(spec)
    spec_res.set_curve_arrays(curve_v, curve_f)
    spec_res.save()

    return redirect('spectral_masw:spectra_list')


# @login_required
def spectra_list_view(request):
    """Вкладка 2: список всех результатов спектрального анализа."""
    results = SpectralResult.objects.select_related('segy_file').all()
    return render(request, 'spectral_masw/spectra_list.html', {'results': results})


# @login_required
def spectra_plot_view(request, pk):
    """Отрисовка выбранного результата (шаг 9–10)."""
    res = get_object_or_404(SpectralResult, pk=pk)

    spec = res.get_spectrogram()
    v_min, v_max = res.extent_x_min, res.extent_x_max
    f_min, f_max = res.extent_y_min, res.extent_y_max

    curve_v = res.get_curve_x()
    curve_f = res.get_curve_y()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    im = ax.imshow(
        spec,
        origin='lower',
        aspect='auto',
        extent=[f_min, f_max, v_min, v_max],
        cmap='viridis',
    )
    ax.plot(curve_v, curve_f, 'r-', linewidth=2, label='Извлечённая кривая')

    ax.set_ylabel('Скорость (м/с)')
    ax.set_xlabel('Частота (Гц)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Амплитуда')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return render(request, 'spectral_masw/spectra_plot.html', {
        'result': res,
        'image_base64': img_b64,
    })



