# Generated by Django 5.2.1 on 2025-05-17 09:31

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Patent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=500, verbose_name='专利标题')),
                ('abstract', models.TextField(verbose_name='摘要')),
                ('description', models.TextField(blank=True, null=True, verbose_name='说明书')),
                ('claims', models.TextField(blank=True, null=True, verbose_name='权利要求')),
                ('publication_date', models.DateField(null=True, verbose_name='公开日')),
                ('application_date', models.DateField(null=True, verbose_name='申请日')),
                ('patent_id', models.CharField(max_length=50, unique=True, verbose_name='专利号')),
            ],
            options={
                'verbose_name': '专利',
                'verbose_name_plural': '专利',
            },
        ),
    ]
