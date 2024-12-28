import pytest

from ..context import utils

curry = utils.curry


class TestCurry:
    def test_simple_function(self):
        @curry
        def test(a, b):
            """
            test docstring, make sure your IDE shows this docstring
            when you hover over the curried function!
            """
            return a, b

        assert test(1)(2) == (1, 2)  # type: ignore
        assert test(1, 2) == (1, 2)
        assert test(1, b=2) == (1, 2)
        assert test(1)(b=2) == (1, 2)  # type: ignore
        assert test(a=1, b=2) == (1, 2)
        assert test(a=1)(b=2) == (1, 2)  # type: ignore
        assert test(b=2)(a=1) == (1, 2)  # type: ignore

    def test_positional_and_keyword_parameters(self):
        @curry
        def test(name, greeting="Hello"):
            return name, greeting

        assert test("Alice") == ("Alice", "Hello")
        assert test("Alice", "Hi") == ("Alice", "Hi")
        assert test(name="Alice") == ("Alice", "Hello")
        assert test(name="Alice", greeting="Hi") == ("Alice", "Hi")
        assert test(greeting="Hi")(name="Alice") == ("Alice", "Hi")  # type: ignore

    def test_variable_positional_arguments(self):
        @curry
        def test(*args):
            return args

        assert test() == ()
        assert test(1) == (1,)
        assert test(1, 2) == (1, 2)
        assert test(1, 2, 3) == (1, 2, 3)

    def test_variable_keyword_arguments(self):
        @curry
        def test(**kwargs):
            return kwargs

        assert test() == {}
        assert test(a=1) == {"a": 1}
        assert test(a=1, b=2) == {"a": 1, "b": 2}
        assert test(a=1, b=2, c=3) == {"a": 1, "b": 2, "c": 3}

    def test_combination_of_positional_keyword_and_args_kwargs(self):
        @curry
        def test(a, b, *args, c=10, **kwargs):
            return a, b, args, c, kwargs

        assert test(1, 2) == (1, 2, (), 10, {})
        assert test(1)(2) == (1, 2, (), 10, {})  # type: ignore
        assert test(1, 2, 3) == (1, 2, (3,), 10, {})
        assert test(1)(2, 3) == (1, 2, (3,), 10, {})  # type: ignore
        assert test(1, 2, 3, 4) == (1, 2, (3, 4), 10, {})
        assert test(1)(2, 3, 4) == (1, 2, (3, 4), 10, {})  # type: ignore
        assert test(1, 2, c=3) == (1, 2, (), 3, {})
        assert test(1)(2, c=3) == (1, 2, (), 3, {})  # type: ignore
        assert test(1, c=3)(2) == (1, 2, (), 3, {})  # type: ignore
        assert test(c=3)(1, 2) == (1, 2, (), 3, {})  # type: ignore
        assert test(c=3)(1)(2) == (1, 2, (), 3, {})  # type: ignore
        assert test(c=3)(1, 2, 9, 8) == (1, 2, (9, 8), 3, {})  # type: ignore
        assert test(c=3)(1)(2, 9, 8) == (1, 2, (9, 8), 3, {})  # type: ignore
        assert test(1, 2, c=3, d=4) == (1, 2, (), 3, {"d": 4})
        assert test(1)(2, c=3, d=4) == (1, 2, (), 3, {"d": 4})  # type: ignore
        assert test(1, c=3)(2, d=4) == (1, 2, (), 3, {"d": 4})  # type: ignore
        assert test(1, c=3, d=4)(2) == (1, 2, (), 3, {"d": 4})  # type: ignore
        assert test(1, 2, 3, c=4, d=5, e=12) == (1, 2, (3,), 4, {"d": 5, "e": 12})

        # ensure duplicate value error is raised
        with pytest.raises(TypeError):
            test(1, c=4)(2, c=3)  # type: ignore
        # passing 'a' as both positional and keyword argument
        with pytest.raises(TypeError):
            test(1, a=2)  # type: ignore
        with pytest.raises(TypeError):
            test(a=1)(1)  # type: ignore

    def test_positional_only_parameters(self):
        @curry
        def test(a, b, /, c):
            return a, b, c

        assert test(1, 2, 3) == (1, 2, 3)  # type: ignore
        assert test(1)(2, 3) == (1, 2, 3)  # type: ignore
        assert test(1)(2)(3) == (1, 2, 3)  # type: ignore

        # assert that error is raised when a and b are passed as keyword arguments
        with pytest.raises(TypeError):
            test(a=1, b=2, c=3)  # type: ignore
        with pytest.raises(TypeError):
            test(1)(b=2, c=3)  # type: ignore

    def test_keyword_only_parameters(self):
        @curry
        def test(*, a, b=10):
            return a, b

        assert test(a=1) == (1, 10)
        assert test(a=1, b=2) == (1, 2)
        assert test(b=2)(a=1) == (1, 2)  # type: ignore

        # assert that error is raised when a is passed as positional argument
        with pytest.raises(TypeError):
            test(1)  # type: ignore
        with pytest.raises(TypeError):
            test(b=10)(2)  # type: ignore

    def test_default_values(self):
        @curry
        def test(a=1, b=2, c=3):
            return a, b, c

        assert test() == (1, 2, 3)
        assert test(4) == (4, 2, 3)
        assert test(4, 5) == (4, 5, 3)
        assert test(4, 5, 6) == (4, 5, 6)

    def test_no_parameters(self):
        @curry
        def test():
            return "No parameters here!"

        assert test() == "No parameters here!"

        with pytest.raises(TypeError):
            test(1)  # type: ignore

    def test_function_with_complex_signature(self):
        @curry
        def test(a, /, b, *, c=10, **kwargs):
            return a, b, c, kwargs

        assert test(1, 2) == (1, 2, 10, {})
        assert test(1)(2) == (1, 2, 10, {})  # type: ignore
        assert test(1, 2, c=3) == (1, 2, 3, {})
        assert test(1)(2, c=3) == (1, 2, 3, {})  # type: ignore
        assert test(1, c=3)(2) == (1, 2, 3, {})  # type: ignore
        assert test(1, 2, d=4) == (1, 2, 10, {"d": 4})
        assert test(1)(2, d=4) == (1, 2, 10, {"d": 4})  # type: ignore
        assert test(1, d=4)(2) == (1, 2, 10, {"d": 4})  # type: ignore

    def test_class_method(self):
        class Test:
            @classmethod
            @curry
            def test(cls, a, b, c=3):
                return a, b, c

        assert Test.test(1)(2) == (1, 2, 3)  # type: ignore
        assert Test.test(1, 2) == (1, 2, 3)
        assert Test.test(1, 2, 4) == (1, 2, 4)
        assert Test.test(1)(2, c=4) == (1, 2, 4)  # type: ignore
        assert Test.test(1, c=4)(2) == (1, 2, 4)  # type: ignore
        assert Test.test(1, c=4)(b=2) == (1, 2, 4)  # type: ignore

    def test_static_method(self):
        class Test:
            @curry
            @staticmethod
            def test(a, b, c=3):
                return a, b, c

        assert Test.test(1)(2) == (1, 2, 3)  # type: ignore
        assert Test.test(1, 2) == (1, 2, 3)
        assert Test.test(1, 2, 4) == (1, 2, 4)
        assert Test.test(1)(2, c=4) == (1, 2, 4)  # type: ignore
        assert Test.test(1, c=4)(2) == (1, 2, 4)  # type: ignore
        assert Test.test(1, c=4)(b=2) == (1, 2, 4)  # type: ignore

    def test_instance_method(self):
        class Test:
            @curry
            def test(self, a, b, c=3):
                return a, b, c

        t = Test()
        assert t.test(1)(2) == (1, 2, 3)  # type: ignore
        assert t.test(1, 2) == (1, 2, 3)
        assert t.test(1, 2, 4) == (1, 2, 4)
        assert t.test(1)(2, c=4) == (1, 2, 4)  # type: ignore
        assert t.test(1, c=4)(2) == (1, 2, 4)  # type: ignore
        assert t.test(1, c=4)(b=2) == (1, 2, 4)  # type: ignore
