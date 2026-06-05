"""User-facing forms for the movies app."""

import re

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    username = forms.CharField(
        min_length=5,
        max_length=30,
        required=True,
        help_text="Minimal 5 karakter. Hanya huruf, angka, dan underscore (_).",
        error_messages={
            "min_length": "Username minimal 5 karakter.",
            "max_length": "Username maksimal 30 karakter.",
            "required": "Username wajib diisi.",
        },
    )
    email = forms.EmailField(
        required=True,
        help_text="Masukkan alamat email yang valid.",
        error_messages={
            "required": "Email wajib diisi.",
            "invalid": "Format email tidak valid.",
        },
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def clean_username(self):
        username = self.cleaned_data.get("username", "")
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise forms.ValidationError(
                "Username hanya boleh mengandung huruf, angka, dan underscore (_)."
            )
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("Username sudah digunakan.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get("email", "").strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("Email sudah terdaftar.")
        return email


class LoginForm(AuthenticationForm):
    pass
