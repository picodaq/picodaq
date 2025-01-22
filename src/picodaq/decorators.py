# This technique taken from python-quantities
# Please visit https://github.com/python-quantities


class with_doc:
    """
    Combine the docstrings of the provided and decorated objects
    to produce the final docstring for the decorated object.
    """

    def __init__(self, base_method):
        self.base_method = base_method

    def __call__(self, new_method):
        new_doc = new_method.__doc__
        base_doc = self.base_method.__doc__
        if base_doc and new_doc:
            new_method.__doc__ = base_doc.strip() + "\n\n        " + new_doc.strip()
        elif base_doc:
            new_method.__doc__ = base_doc
        return new_method        
