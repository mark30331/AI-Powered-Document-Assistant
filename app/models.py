from django.db import models


class Document(models.Model):
    session_key = models.CharField(max_length=40, db_index=True)
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.filename


class DocumentChunk(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    page_number = models.PositiveIntegerField()
    content = models.TextField()

    class Meta:
        indexes = [
            models.Index(fields=["document", "page_number"]),
        ]
