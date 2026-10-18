"""Offline sorting adapters and the template bank consumed online."""
from .templates import TemplateBank, build_template_bank, estimate_noise, windowed_filter_error

__all__ = ["TemplateBank", "build_template_bank", "estimate_noise", "windowed_filter_error"]
