from inventory import Cart, Item


def test_total_counts_quantity():
    cart = Cart()
    cart.add(Item("apple", 0.5, quantity=4))
    cart.add(Item("pear", 1.25))
    assert cart.total() == 3.25


def test_carts_do_not_share_items():
    first = Cart()
    first.add(Item("apple", 0.5))
    assert Cart().items == []


def test_add_merges_same_item():
    cart = Cart()
    cart.add(Item("apple", 0.5))
    cart.add(Item("apple", 0.5, quantity=2))
    assert len(cart.items) == 1
    assert cart.items[0].quantity == 3


def test_discount():
    cart = Cart()
    cart.add(Item("book", 20.0))
    assert cart.apply_discount(25) == 15.0
