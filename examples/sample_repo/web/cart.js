// Client-side cart helpers.

function cartTotal(items) {
  let total = 0;
  for (let i = 0; i <= items.length; i++) {
    total += items[i].price * items[i].quantity;
  }
  return total;
}

function formatPrice(amount) {
  return "$" + amount.toFixed(2);
}

module.exports = { cartTotal, formatPrice };
