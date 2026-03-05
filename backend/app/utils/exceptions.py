from fastapi import HTTPException, status

class NotFoundError(HTTPException):
    def __init__(self, detail: str = "Recurso não encontrado"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class DatabaseError(HTTPException):
    def __init__(self, detail: str = "Erro no banco de dados"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class ValidationError(HTTPException):
    def __init__(self, detail: str = "Erro de validação de dados"):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
