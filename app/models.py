from django.db import models
from django.utils import timezone

import pandas as pd
import numpy as np
import time
import turicreate as tc

# Create your models here.


class LogMessage(models.Model):
    message = models.CharField(max_length=300)
    log_date = models.DateTimeField("date logged")

    def __str__(self):
        """Returns a string representation of a message."""
        date = timezone.localtime(self.log_date)
        return f"'{self.message}' logged on {date.strftime('%A, %d %B, %Y at %X')}"

class ItemRecommender:

    def itemRecommender(self):
        print('Item Recommender Getting here 1')
        return ''
