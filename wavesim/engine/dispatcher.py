import inspect
from typing import Callable, Optional
from types import UnionType


class _Dispatcher:
    """
    Helper class to define a series of multiple dispatch functions (``_mix``, ``_multiply``, etc.)
    on the Array object.

    Attributes:
        functions (dict): A dictionary to store functions with their type signatures.
    """

    def __init__(self, function_name: str):
        """
        Initialize a Dispatcher for a new function. Don't call directly, use the @dispatch decorator instead.

        Args:
            function_name (str): The name of the function to dispatch.
        """
        self.functions = {}
        self.signature = None
        self.keywords = ()
        self.name = function_name
        if function_name in _dispatchers:
            raise KeyError(f"A dispatcher for  '{function_name}' already exists.")
        _dispatchers[function_name] = self

    def register(self, func: Callable):
        """
        Register a function with the dispatcher.

        All arguments must either be positional or keyword only (use ``, /, *`` in the signature).
        The function must have the same number of positional arguments as any previously registered function,
        and the keyword-only arguments must have the same names.

        Args:
            func (Callable): The function to register.

        Raises:
            ValueError: If a function with the same signature is already registered,
                or if the signature is incompatible with that of a previously registered function.
        """

        signature = inspect.signature(func)
        arguments = list(signature.parameters.values())

        # Check if all arguments are either positional or keyword only
        if any(p.kind != p.POSITIONAL_ONLY and p.kind != p.KEYWORD_ONLY for p in signature.parameters.values()):
            raise ValueError(f"All arguments must either be position only or keyword only\n{_line(func)}")

        if self.signature is None:
            self.signature = signature
            self.keywords = tuple(p.name for p in signature.parameters.values() if p.kind == p.KEYWORD_ONLY)
        else:
            # If a signature was already registered, check if the new signature is compatible
            if len(arguments) != len(self.signature.parameters) or any(
                p1.kind != p2.kind or p1.name != p2.name
                for p1, p2 in zip(self.signature.parameters.values(), arguments)
            ):
                raise ValueError(
                    f"Function signature or argument names do not match previously registered function.\n"
                    f"First definition {_line(func)}\n"
                    f"Previous definition {_line(next(iter(self.functions.values())))}"
                )
        previous = self._register(arguments, func)

        if previous is not None:
            def1 = previous.__code__  # noqa
            def2 = func.__code__  # noqa
            raise ValueError(
                f"""
                Function with signature {arguments} already registered.
                First definition {def1.co_filename}:{def1.co_firstlineno}
                Second definition {def2.co_filename}:{def2.co_firstlineno}."""
            )

    def _register(self, arguments: list, func: Callable) -> Optional[Callable]:
        """Recursively register a function taking into account union types
        If at least one variant of the function was newly registered, return None
        Otherwise, return (one of) the function(s) that was already registered
        """
        previous = True  # placeholder to store the first function that was already registered
        for i, p in enumerate(arguments):
            if isinstance(p.annotation, UnionType):
                for utype in p.annotation.__args__:
                    arguments[i] = p.replace(annotation=utype)
                    prev = self._register(arguments, func)
                    if previous is not None:
                        previous = prev  # overwrite previous with None or the function that was already registered
                arguments[i] = p
                return previous if previous is not True else None  # only loop over the first non-UnionType parameter

        # Construct a hashable signature object from the types of the arguments
        # Supports generic types using the __origin__ attribute
        signature_key = tuple(p.annotation for p in arguments)
        signature_key = tuple(getattr(t, "__origin__", t) for t in signature_key)

        if signature_key not in self.functions:
            self.functions[signature_key] = func
        else:
            return self.functions[signature_key]  # return already registered function

    def __call__(self, *args, **kwargs):
        """Look up the matching function for this set of argument types and call it.

        The lookup is based on the types of the arguments, not the values.
        If the signature is not in the functions dict, ``lookup`` is called to find the closest match, and
        the match is cached in the dict for future calls.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the called function.

        Raises:
            NotImplementedError: If no function is found for the given signature.
        """
        try:
            signature_key = tuple(type(arg) for arg in args) + tuple(kwargs[keyword] for keyword in self.keywords)
        except KeyError as e:
            raise ValueError(f"Missing keyword argument {e}")
        try:
            func = self.functions[signature_key]
        except KeyError:
            func = self.lookup(*args, **kwargs)
        return func(*args, **kwargs)

    def print(self):
        """Print the function signatures."""
        print(f"Function {self.name}")
        for sig, func in self.functions.items():
            arguments = ", ".join(arg.__name__ for arg in sig)
            print(f"{self.name}({arguments}) -> {func.__code__.co_filename}:{func.__code__.co_firstlineno}")

    def lookup(self, *args, **kwargs) -> Callable:
        """
        Lookup the most specific function for the given signature.

        This function scans all registered functions to see if they match the signature in the sense
        that all arguments of the signature are ``issubclass`` of the arguments of the registered function's signature.

        The first function that matches is considered a 'candidate'. If a new candidate is found that is more specific
        (in the sense that one or more of the arguments are subclasses of those of the candidate),
        the candidate is replaced. The final remaining candidate is called, and the match is cached
        in the function's dict.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            return value from the called function

        Raises:
            NotImplementedError: If no function is found for the given signature.
        """
        candidate = None
        binding = self.signature.bind(*args, **kwargs)  # throws if arguments don't match the signature
        for sig_key, func in self.functions.items():
            if all(isinstance(v, t) for v, t in zip(binding.arguments.values(), sig_key)):
                # found a match
                if candidate is None or all(issubclass(t, s) for t, s in zip(sig_key, candidate[0])):
                    # the match is more specific than we had until now, replace the candidate
                    candidate = (sig_key, func)

        if candidate is None:
            # No candidate was found, print all function signatures for debugging and raise an error
            self.print()
            arguments = [type(arg) for arg in args]
            keyword_arguments = {name: type(value) for name, value in kwargs.items()}
            raise NotImplementedError(
                f"No function {self.name} found for signature {arguments}, /, *, {keyword_arguments}"
            )

        # also register the function with this specific argument types, so that we don't need to look it up again
        self.functions[candidate[0]] = candidate[1]
        return candidate[1]


def dispatch(func: Callable) -> _Dispatcher:
    """Decorator to register a function with the dispatcher.

    The decorated function can be called normally, and the dispatcher
    will take care of calling the correct function under the hood.
    """
    try:
        dispatcher = _dispatchers[func.__name__]
    except KeyError:
        dispatcher = _Dispatcher(func.__name__)
    dispatcher.register(func)
    return dispatcher


_dispatchers = {}


def _line(f: Callable) -> str:
    return f"{f.__code__.co_filename}:{f.__code__.co_firstlineno}"  # noqa
