from django.db import models

class Patent(models.Model):
    """专利数据模型"""
    title = models.CharField(max_length=500, verbose_name='专利标题')
    abstract = models.TextField(verbose_name='摘要')
    description = models.TextField(verbose_name='说明书', null=True, blank=True)
    claims = models.TextField(verbose_name='权利要求', null=True, blank=True)
    publication_date = models.DateField(verbose_name='公开日', null=True)
    application_date = models.DateField(verbose_name='申请日', null=True)
    patent_id = models.CharField(max_length=50, unique=True, verbose_name='专利号')
    
    class Meta:
        verbose_name = '专利'
        verbose_name_plural = '专利'
        
    def __str__(self):
        return self.title 