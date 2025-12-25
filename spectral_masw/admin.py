# spectral_masw/admin.py
from django.contrib import admin
from .models import SegyFile, TraceHeader, SpectralResult


@admin.register(SegyFile)
class SegyFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'upload_date', 'num_traces', 'num_samples', 'sample_rate')
    list_filter = ('upload_date',)
    search_fields = ('name',)
    readonly_fields = ('upload_date', 'num_traces', 'num_samples')

    fieldsets = (
        ('Основная информация', {
            'fields': ('name', 'original_file', 'upload_date'),
        }),
        ('Параметры', {
            'fields': ('sample_rate', 'num_traces', 'num_samples'),
        }),
        ('Данные', {
            'fields': ('trace_data',),
            'classes': ('collapse',),
        }),
    )


@admin.register(TraceHeader)
class TraceHeaderAdmin(admin.ModelAdmin):
    list_display = ('id', 'segy_file', 'trace_number')
    list_filter = ('segy_file',)
    search_fields = ('segy_file__name', 'trace_number')

    readonly_fields = ('segy_file', 'trace_number')

    fieldsets = (
        ('Связь', {'fields': ('segy_file', 'trace_number')}),
        # Добавь сюда остальные поля заголовка, если они есть
    )

    def has_add_permission(self, request):
        # Заголовки создаются автоматически при парсинге SEGY
        return False


@admin.register(SpectralResult)
class SpectralResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'segy_file', 'created_at')
    search_fields = ('name', 'segy_file__name')

    # ВНИМАНИЕ: здесь только существующие поля!
    readonly_fields = (
        'created_at',
        'segy_file',
        'spectrogram_matrix',
        'curve_x_array',
        'curve_y_array',
    )

    fieldsets = (
        ('Общее', {
            'fields': ('name', 'segy_file', 'created_at'),
        }),
        ('Пределы осей', {
            'fields': ('extent_x_min', 'extent_x_max', 'extent_y_min', 'extent_y_max'),
        }),
        ('Данные', {
            'fields': ('spectrogram_matrix', 'curve_x_array', 'curve_y_array'),
            'classes': ('collapse',),
        }),
        ('Прочее', {
            'fields': ('processing_notes',),
        }),
    )

    def has_add_permission(self, request):
        # Результаты создаются через обработку, а не руками
        return False
