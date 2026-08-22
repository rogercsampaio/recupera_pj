# config/network.py
import socket
from loguru import logger

if not hasattr(socket, "_original_getaddrinfo"):
    socket._original_getaddrinfo = socket.getaddrinfo


def configurar_ipv4():
    if getattr(socket, "_ipv4_configurado", False):
        return

    original_getaddrinfo = socket._original_getaddrinfo

    def getaddrinfo_ipv4(*args, **kwargs):
        resultados = original_getaddrinfo(*args, **kwargs)

        resultados_ipv4 = [
            resultado
            for resultado in resultados
            if resultado[0] == socket.AF_INET
        ]

        return resultados_ipv4

    socket.getaddrinfo = getaddrinfo_ipv4
    socket._ipv4_configurado = True
    
    logger.info("Python configurado para priorizar IPv4.")