class PdfParseError(ValueError):
    pass


class EmbeddingError(ValueError):
    pass


class JobNotFoundError(LookupError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(job_id)


class DocumentFetchError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass
