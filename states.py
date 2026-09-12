import os

# typed DICT 

from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : str

# pydantic approach

from pydantic import BaseModel , field_validator

class State(BaseModel):
    topic : str
    score : int
    summary : str = ''

    @field_validator
    def score_positive(cls , v):
        if v < 0:
            raise ValueError('Score must be positive')

# python dataclasses

from dataclasses import dataclass , field

@dataclass
class State:
    topic : str = ''
    summary : str = ''
    messages : list = field(default_factory=list)
