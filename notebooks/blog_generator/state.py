from pydantic import BaseModel

class BlogState(BaseModel):
    ## User Input
    topic:str=""
    audience:str=""
    ##R

     