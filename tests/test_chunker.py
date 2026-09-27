from repo_debug_agent.chunker import chunk_source, mask_code
from repo_debug_agent.languages import BY_NAME


def names(chunks):
    return [(c.kind, c.name, c.start_line, c.end_line) for c in chunks]


def test_python_functions_methods_decorators_and_module_code():
    src = '''import os

LIMIT = 3


@cache
def top(a):
    return a


class Cart:
    """Doc."""

    tax = 0.1

    def __init__(self):
        self.items = []

    @property
    async def total(self):
        def helper():
            return 1

        return helper()


if __name__ == "__main__":
    top(1)
'''
    chunks = chunk_source("m.py", src, BY_NAME["python"])
    assert names(chunks) == [
        ("module", "<module>", 1, 3),
        ("function", "top", 6, 8),
        ("class", "Cart", 11, 14),
        ("method", "Cart.__init__", 16, 17),
        ("method", "Cart.total", 19, 24),
        ("module", "<module>", 27, 28),
    ]
    cart = next(c for c in chunks if c.name == "Cart")
    assert cart.body_end == 24  # the whole class, for fixes that must rewrite it
    total = next(c for c in chunks if c.name == "Cart.total")
    assert total.code.startswith("    @property")
    assert [c.key for c in chunks if c.name == "<module>"] == ["m.py::<module>", "m.py::<module>#2"]


def test_python_syntax_error_falls_back_to_whole_file():
    chunks = chunk_source("bad.py", "def f(:\n    pass\n", BY_NAME["python"])
    assert names(chunks) == [("module", "<module>", 1, 2)]


def test_mask_code_blanks_strings_and_comments_but_keeps_layout():
    js = BY_NAME["javascript"]
    src = "const a = \"x{y}\"; // {\n/* { */ let b = `t\n{`; c = '{';"
    masked = mask_code(src, js)
    assert len(masked) == len(src)
    assert masked.count("\n") == src.count("\n")
    assert "{" not in masked and "}" not in masked


def test_mask_code_keeps_rust_lifetimes():
    rust = BY_NAME["rust"]
    masked = mask_code("fn f<'a>(x: &'a str) -> char { '{' }", rust)
    assert masked.count("{") == 1 and masked.count("}") == 1


def test_javascript_functions_classes_and_arrows():
    src = """import x from "y";

export function add(a, b) {
  const s = "}";
  return a + b;
}

const mul = (a, b) => {
  return a * b;
};

class Cart extends Base {
  constructor() {
    super();
  }

  get total() {
    return 1;
  }
}

module.exports = { add, mul };
"""
    chunks = chunk_source("a.js", src, BY_NAME["javascript"])
    assert names(chunks) == [
        ("module", "<module>", 1, 1),
        ("function", "add", 3, 6),
        ("function", "mul", 8, 10),
        ("class", "Cart", 12, 12),
        ("method", "Cart.constructor", 13, 15),
        ("method", "Cart.total", 17, 19),
        ("module", "<module>", 22, 22),
    ]


def test_go_methods_get_receiver_names():
    src = """package main

type Stack struct {
\titems []int
}

func (s *Stack) Push(v int) {
\ts.items = append(s.items, v)
}
"""
    chunks = chunk_source("s.go", src, BY_NAME["go"])
    assert ("class", "Stack", 3, 5) in names(chunks)
    assert ("method", "Stack.Push", 7, 9) in names(chunks)


def test_c_function_returning_struct_pointer_and_allman_braces():
    src = """#include <stdio.h>

struct node *make_node(int v)
{
    return NULL;
}
"""
    chunks = chunk_source("n.c", src, BY_NAME["c"])
    assert ("function", "make_node", 3, 6) in names(chunks)


def test_java_annotations_belong_to_the_method():
    src = """public class Calc {
    @Override
    public String toString() {
        return "{";
    }
}
"""
    chunks = chunk_source("Calc.java", src, BY_NAME["java"])
    assert ("method", "Calc.toString", 2, 5) in names(chunks)


def test_ruby_defs_and_endless_methods():
    src = """class Greeter
  def initialize(name)
    @name = name
  end

  def self.hello = "hi"

  def greet
    if @name
      puts "hi"
    end
  end
end
"""
    chunks = chunk_source("g.rb", src, BY_NAME["ruby"])
    assert names(chunks) == [
        ("class", "Greeter", 1, 1),
        ("method", "Greeter.initialize", 2, 4),
        ("method", "Greeter.hello", 6, 6),
        ("method", "Greeter.greet", 8, 12),
    ]


def test_long_module_code_is_split():
    src = "\n".join(f"x{i} = {i}" for i in range(300))
    chunks = chunk_source("big.py", src, BY_NAME["python"])
    assert len(chunks) == 3
    assert all(c.span <= 120 for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 300
