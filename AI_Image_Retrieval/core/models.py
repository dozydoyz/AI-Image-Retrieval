from django.db import models
from django.contrib.auth.models import User
#这里定义搜索和历史记录模型
class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="用户")
    image = models.ImageField(upload_to='search_uploads/', verbose_name="上传图片")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="搜索时间")

    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        ordering = ['-created_at']
        verbose_name = "搜索记录"
        verbose_name_plural = verbose_name