from app.utils.exceptions import ValidationError

def validar_preco(preco: float):
    if preco < 0:
        raise ValidationError("O preço não pode ser negativo")
    return True

def validar_id_jogo(jogo_id: int):
    if jogo_id <= 0:
        raise ValidationError("ID de jogo inválido")
    return True
