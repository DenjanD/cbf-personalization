# myapp/templatetags/custom_filters.py

from django import template

register = template.Library()

@register.filter
def seconds_to_hhmmss(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'
