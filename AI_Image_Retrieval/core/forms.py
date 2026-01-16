from django import forms
from .models import SearchHistory

class UploadForm(forms.ModelForm):
    class Meta:
        model = SearchHistory
        fields = ['image']
        labels = {
            'image': '选择一张图片开始搜索'
        }