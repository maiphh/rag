from abc import ABC, abstractmethod
from enum_manager import *
from typing import Any, Optional

class Router(ABC):
    @abstractmethod
    def route(self, query: Optional[Any] = None):
        pass

class ManualDomainRouter(Router):
    def __init__(self, domain = DOMAIN.ALL.value):
        self.domain = domain

    def set_domain(self, domain: DOMAIN):
        self.domain = domain.value

    def route(self, query: Optional[Any] = None):
        return [self.domain]
    
class KeywordRouter(Router):
    def __init__(self, keyword_map: dict[DOMAIN, list[str]], default_domain=DOMAIN.ALL.value):
        self.keyword_map = keyword_map
        self.default_domain = default_domain

    def route(self, query: Optional[str] = None):
        if not query:
            return self.default_domain
        
        query_lower = query.lower()
        for domain, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    return domain
        return self.default_domain