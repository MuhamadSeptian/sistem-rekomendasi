from django import template

register = template.Library()


@register.filter
def is_rating(user_rating, value):
    """Check if user_rating exists and matches the given value.
    Usage: {% if user_rating|is_rating:5 %}checked{% endif %}
    """
    if user_rating is None:
        return False
    return user_rating.rating == int(value)
