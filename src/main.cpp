#include <iostream>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <variant>
#include <optional>
#include <numeric>
#include <tuple>
#include <set>

using namespace std;

#ifndef GUARD_DPSG_STRONG_TYPES_HPP
#define GUARD_DPSG_STRONG_TYPES_HPP

#include <type_traits>
#include <utility>

namespace dpsg {

template <class Instance, template <typename...> class Template>
struct is_template_instance : std::false_type {};
template <template <typename...> class C, typename... Args>
struct is_template_instance<C<Args...>, C> : std::true_type {};

template <class Instance, template <typename...> class Template>
constexpr static inline bool is_template_instance_v =
    is_template_instance<Instance, Template>::value;

namespace strong_types {

namespace detail {
template <class...>
struct void_t_impl {
  using type = void;
};
template <class... Ts>
using void_t = typename void_t_impl<Ts...>::type;
template <class T, class = void>
struct has_value : std::false_type {};
template <class T>
struct has_value<T, void_t<decltype(std::declval<T>().value)>>
    : std::true_type {};
}  // namespace detail
template <class T>
using has_value = detail::has_value<T>;
template <class T>
constexpr bool has_value_v = has_value<T>::value;

struct get_value_t {
  template <class T, std::enable_if_t<has_value_v<T>, int> = 0>
  inline constexpr decltype(auto) operator()(T&& t) noexcept {
    return std::forward<T>(t).value;
  }
  template <class T, std::enable_if_t<!has_value_v<T>, int> = 0>
  inline constexpr decltype(auto) operator()(T&& t) noexcept {
    return std::forward<T>(t);
  }
  template <class T, std::enable_if_t<has_value_v<T>, int> = 0>
  inline constexpr auto& operator()(T& t) noexcept {
    return t.value;
  }
  template <class T, std::enable_if_t<!has_value_v<T>, int> = 0>
  inline constexpr auto& operator()(T& t) noexcept {
    return t;
  }
};

template <class To>
struct get_value_then_cast_t {
  template <class T>
  inline constexpr To operator()(T&& t) noexcept {
    return static_cast<To>(get_value_t{}(std::forward<T>(t)));
  }
};

template <class To, class Cl>
struct cast_to_then_construct_t {
  template <class T>
  inline constexpr Cl operator()(T&& t) noexcept {
    return Cl{static_cast<To>(std::forward<T>(t))};
  }
};

struct passthrough_t {
  template <class T>
  inline constexpr decltype(auto) operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }
};

template <class T>
struct construct_t {
  template <class... Ts>
  inline constexpr T operator()(Ts&&... ts) const noexcept {
    return T{std::forward<Ts>(ts)...};
  }
};

namespace black_magic {
template <class... Ts>
struct tuple;

template <class T1, class T2>
struct concat_tuples;
template <class... T1s, class... T2s>
struct concat_tuples<tuple<T1s...>, tuple<T2s...>> {
  using type = tuple<T1s..., T2s...>;
};
template <class T1, class T2>
using concat_tuples_t = typename concat_tuples<T1, T2>::type;

template <class T, class U>
struct for_each;
template <class U, class... Ts>
struct for_each<tuple<Ts...>, U> : U::template type<Ts>... {};

template <template <class...> class S, class... Ts>
struct apply {
  template <class Tuple>
  struct type;
  template <template <class...> class Tuple, class Op, class Target>
  struct type<Tuple<Op, Target>> : S<Op, Target, Ts...> {};
};

struct deduce;
namespace detail {
template <class T, class D, class R, class = void>
struct deduce_return_type_impl {
  using type = T;
};
template <class D, class R>
struct deduce_return_type_impl<deduce, D, R> {
  using type = D;
};
template <class D, class R>
struct deduce_return_type_impl<R, D, R> {
  using type = R;
};

template <template<class ...> class T, class D, class R, class ...Ts>
struct deduce_return_type_impl<T<Ts...>, D, R, std::enable_if_t<!std::is_same<T<Ts...>, R>::value>> {
  using type = T<typename deduce_return_type_impl<Ts, R, R>::type...>;
};

}  // namespace detail

template <class T, class D, class R>
using deduce_return_type = typename detail::deduce_return_type_impl<T, D, R>::type;
}  // namespace black_magic

// clang-tidy off
#define DPSG_DEFINE_BINARY_OPERATOR(name, sym)                             \
  struct name {                                                            \
    template <class T, class U>                                            \
    constexpr inline decltype(auto) operator()(T&& left,                   \
                                               U&& right) const noexcept { \
      return std::forward<T>(left) sym std::forward<U>(right);             \
    }                                                                      \
  };

#define DPSG_DEFINE_UNARY_OPERATOR(name, sym)                          \
  struct name {                                                        \
    template <class U>                                                 \
    constexpr inline decltype(auto) operator()(U&& u) const noexcept { \
      return sym std::forward<U>(u);                                   \
    }                                                                  \
  };

#define DPSG_APPLY_TO_BINARY_OPERATORS(f)                            \
  f(plus, +) f(minus, -) f(divides, /) f(multiplies, *) f(modulo, %) \
      f(equal, ==) f(not_equal, !=) f(lesser, <) f(greater, >)       \
          f(lesser_equal, <=) f(greater_equal, >=) f(binary_or, |)   \
              f(binary_and, &) f(binary_xor, ^) f(shift_right, <<)   \
                  f(shift_left, >>) f(boolean_or, ||) f(boolean_and, &&)

#define DPSG_APPLY_TO_SELF_ASSIGNING_BINARY_OPERATORS(f)                       \
  f(plus_assign, +=) f(minus_assign, -=) f(divides_assign, /=)                 \
      f(multiplies_assign, *=) f(modulo_assign, %=) f(shift_right_assign, <<=) \
          f(shift_left_assign, >>=) f(binary_and_assign, &=)                   \
              f(binary_or_assign, |=) f(binary_xor_assign, ^=)

#define DPSG_APPLY_TO_UNARY_OPERATORS(f)                           \
  f(boolean_not, !) f(binary_not, ~) f(negate, -) f(positivate, +) \
      f(dereference, *) f(address_of, &) f(increment, ++) f(decrement, --)

DPSG_APPLY_TO_BINARY_OPERATORS(DPSG_DEFINE_BINARY_OPERATOR)
DPSG_APPLY_TO_SELF_ASSIGNING_BINARY_OPERATORS(DPSG_DEFINE_BINARY_OPERATOR)
DPSG_APPLY_TO_UNARY_OPERATORS(DPSG_DEFINE_UNARY_OPERATOR)
// clang-tidy on

struct post_increment {
  template <class U>
  constexpr inline decltype(auto) operator()(U& u) const noexcept {
    return u++;
  }
};

struct post_decrement {
  template <class U>
  constexpr inline decltype(auto) operator()(U& u) const noexcept {
    return u--;
  }
};

namespace detail {
template <class Op,
          class Left,
          class Right,
          class Result = passthrough_t,
          class TransformLeft = get_value_t,
          class TransformRight = get_value_t>
struct implement_binary_operation;

template <class Op,
          class Arg,
          class Result = passthrough_t,
          class Transform = get_value_t>
struct implement_unary_operation;

#define DPSG_DEFINE_FRIEND_BINARY_OPERATOR_IMPLEMENTATION(op, sym)           \
  template <class Left,                                                      \
            class Right,                                                     \
            class Result,                                                    \
            class TransformLeft,                                             \
            class TransformRight>                                            \
  struct implement_binary_operation<op,                                      \
                                    Left,                                    \
                                    Right,                                   \
                                    Result,                                  \
                                    TransformLeft,                           \
                                    TransformRight> {                        \
    friend constexpr decltype(auto) operator sym(const Left& left,           \
                                                 const Right& right) {       \
      return Result{}(op{}(TransformLeft{}(left), TransformRight{}(right))); \
    }                                                                        \
    friend constexpr decltype(auto) operator sym(Left& left,                 \
                                                 const Right& right) {       \
      return Result{}(op{}(TransformLeft{}(left), TransformRight{}(right))); \
    }                                                                        \
    friend constexpr decltype(auto) operator sym(const Left& left,           \
                                                 Right& right) {             \
      return Result{}(op{}(TransformLeft{}(left), TransformRight{}(right))); \
    }                                                                        \
    friend constexpr decltype(auto) operator sym(Left& left, Right& right) { \
      return Result{}(op{}(TransformLeft{}(left), TransformRight{}(right))); \
    }                                                                        \
  };

#define DPSG_DEFINE_FRIEND_SELF_ASSIGN_BINARY_OPERATOR_IMPLEMENTATION(op, sym) \
  template <class Left,                                                        \
            class Right,                                                       \
            class Result,                                                      \
            class TransformLeft,                                               \
            class TransformRight>                                              \
  struct implement_binary_operation<op,                                        \
                                    Left,                                      \
                                    Right,                                     \
                                    Result,                                    \
                                    TransformLeft,                             \
                                    TransformRight> {                          \
    template <                                                                 \
        class T,                                                               \
        std::enable_if_t<std::is_same<std::decay_t<T>, Left>::value, int> = 0> \
    friend constexpr decltype(auto) operator sym(T& left,                      \
                                                 const Right& right) {         \
      op{}(TransformLeft{}(left), TransformRight{}(right));                    \
      return left;                                                             \
    }                                                                          \
  };

#define DPSG_DEFINE_FRIEND_UNARY_OPERATOR_IMPLEMENTATION(op, sym)             \
  template <class Arg, class Result, class Transform>                         \
  struct implement_unary_operation<op, Arg, Result, Transform> {              \
    template <                                                                \
        class T,                                                              \
        std::enable_if_t<std::is_same<std::decay_t<T>, Arg>::value, int> = 0> \
    friend constexpr decltype(auto) operator sym(T&& arg) {                   \
      return Result{}(op{}(Transform{}(std::forward<T>(arg))));               \
    }                                                                         \
  };

template <class L, class R, class TL>
struct implement_unary_operation<post_increment, L, R, TL> {
  template <class T,
            std::enable_if_t<std::is_same<std::decay_t<T>, L>::value, int> = 0>
  friend constexpr decltype(auto) operator++(T& left, int) {
    return R{}(post_increment{}(TL{}(left)));
  }
};

template <class L, class R, class TL>
struct implement_unary_operation<post_decrement, L, R, TL> {
  template <class T,
            std::enable_if_t<std::is_same<std::decay_t<T>, L>::value, int> = 0>
  friend constexpr decltype(auto) operator--(T& left, int) {
    return R{}(post_decrement{}(TL{}(left)));
  }
};

DPSG_APPLY_TO_BINARY_OPERATORS(
    DPSG_DEFINE_FRIEND_BINARY_OPERATOR_IMPLEMENTATION)
DPSG_APPLY_TO_SELF_ASSIGNING_BINARY_OPERATORS(
    DPSG_DEFINE_FRIEND_SELF_ASSIGN_BINARY_OPERATOR_IMPLEMENTATION)
DPSG_APPLY_TO_UNARY_OPERATORS(DPSG_DEFINE_FRIEND_UNARY_OPERATOR_IMPLEMENTATION)

}  // namespace detail

template <class Op,
          class Left,
          class Right,
          class Result = passthrough_t,
          class TransformLeft = get_value_t,
          class TransformRight = get_value_t>
using implement_binary_operation = detail::implement_binary_operation<
    Op,
    Left,
    Right,
    Result,
    TransformLeft,
    TransformRight>;

template <class Op,
          class Arg,
          class Result = passthrough_t,
          class Transform = get_value_t>
using implement_unary_operation = detail::implement_unary_operation<
    Op,
    Arg,
    Result,
    Transform>;

template <class Operation,
          class Arg,
          class Result = construct_t<Arg>,
          class Transform = get_value_t>
struct implement_symmetric_operation
    : implement_binary_operation<
          Operation,
          Arg,
          Arg,
          Result,
          Transform,
          Transform> {};

template <class Operation,
          class Left,
          class Right,
          class Return = passthrough_t,
          class TransformLeft = get_value_t,
          class TransformRight = get_value_t>
struct implement_commutative_operation
    : implement_binary_operation<Operation,
                                 Left,
                                 Right,
                                 Return,
                                 TransformLeft,
                                 TransformRight>,
      implement_binary_operation<Operation,
                                 Right,
                                 Left,
                                 Return,
                                 TransformRight,
                                 TransformLeft> {};

using comparison_operators = black_magic::
    tuple<equal, not_equal, lesser_equal, greater_equal, lesser, greater>;

using unary_boolean_operators = black_magic::tuple<boolean_not>;
using binary_boolean_operators = black_magic::tuple<boolean_and, boolean_or>;
using boolean_operators =
    black_magic::concat_tuples_t<unary_boolean_operators,
                                 binary_boolean_operators>;

using unary_bitwise_operators = black_magic::tuple<binary_not>;
using binary_bitwise_operators = black_magic::tuple<binary_and,
                                                    binary_or,
                                                    binary_xor,
                                                    shift_left,
                                                    shift_right,
                                                    shift_left_assign,
                                                    shift_right_assign>;
using bitwise_operators =
    black_magic::concat_tuples_t<unary_bitwise_operators,
                                 binary_bitwise_operators>;

using unary_arithmetic_operators = black_magic::tuple<negate,
                                                      positivate,
                                                      increment,
                                                      decrement,
                                                      post_increment,
                                                      post_decrement>;
using binary_arithmetic_operators = black_magic::tuple<plus,
                                                       minus,
                                                       multiplies,
                                                       divides,
                                                       modulo,
                                                       plus_assign,
                                                       minus_assign,
                                                       multiplies_assign,
                                                       divides_assign,
                                                       modulo_assign>;
using arithmetic_operators =
    black_magic::concat_tuples_t<unary_arithmetic_operators,
                                 binary_arithmetic_operators>;

template <class Arg1,
          class Arg2,
          class R,
          class T1 = get_value_t,
          class T2 = get_value_t>
struct make_commutative_operator {
  template <class Op>
  using type = implement_commutative_operation<Op, Arg1, Arg2, R, T1, T2>;
};
template <class Arg, class R = black_magic::deduce, class T = get_value_t>
struct make_symmetric_operator {
  template <class Op>
  using type = implement_symmetric_operation<Op, Arg, R, T>;
};
template <class Arg, class R = black_magic::deduce, class T = get_value_t>
struct make_unary_operator {
  template <class Op>
  using type = implement_unary_operation<Op, Arg, R, T>;
};
template <class Left,
          class Right,
          class R = black_magic::deduce,
          class TL = get_value_t,
          class TR = get_value_t>
struct make_binary_operator {
  template <class Op>
  using type = implement_binary_operation<Op, Left, Right, R, TL, TR>;
};

struct comparable {
  template <class Arg>
  struct type
      : black_magic::for_each<
            comparison_operators,
            make_symmetric_operator<
                Arg,
                construct_t<bool> /* passthrough causes rt errors on MSVC */>> {
  };
};

template <class Arg2>
struct comparable_with {
  template <class Arg1>
  struct type
      : black_magic::for_each<
            comparison_operators,
            make_commutative_operator<
                Arg1,
                Arg2,
                construct_t<bool> /* passthrough causes rt errors on MSVC */>> {
  };
};

struct arithmetic {
  template <class Arg>
  struct type
      : black_magic::for_each<binary_arithmetic_operators,
                              make_symmetric_operator<Arg, construct_t<Arg>>>,
        black_magic::for_each<
            unary_arithmetic_operators,
            make_unary_operator<Arg, construct_t<Arg>, get_value_t>> {};
};

template <class Op,
          class Return = black_magic::deduce,
          class Transform = get_value_t>
struct symmetric {
  template <class T>
  using type =
      implement_binary_operation<Op, T, T, Return, Transform, Transform>;
};

template <class Arg2,
          class R = black_magic::deduce,
          class T1 = get_value_t,
          class T2 = get_value_t>
struct arithmetically_compatible_with {
  template <class Arg1>
  struct type : black_magic::for_each<
                    binary_arithmetic_operators,
                    make_commutative_operator<
                        Arg1,
                        Arg2,
                        black_magic::deduce_return_type<R, construct_t<Arg1>, Arg1>,
                        T1,
                        T2>> {};
};

template <class Op,
          class Arg2,
          class R = black_magic::deduce,
          class T1 = get_value_t,
          class T2 = get_value_t>
struct commutative_under {
  template <class Arg1>
  using type = implement_commutative_operation<
      Op,
      Arg1,
      Arg2,
      black_magic::deduce_return_type<R, construct_t<Arg1>, Arg1>,
      T1,
      T2>;
};

template <class Op,
          class Arg2,
          class R = black_magic::deduce,
          class T1 = get_value_t,
          class T2 = get_value_t>
struct compatible_under {
  template <class Arg1>
  using type = implement_binary_operation<
      Op,
      Arg1,
      Arg2,
      black_magic::deduce_return_type<R, construct_t<Arg1>, Arg1>,
      T1,
      T2>;
};

template <class T, class... Ts>
struct derive_t : Ts::template type<T>... {};

template <class Type, class Tag, class... Params>
struct strong_value : derive_t<strong_value<Type, Tag, Params...>, Params...> {
  using value_type = Type;

  template <
      class U,
      std::enable_if_t<std::is_convertible<std::decay_t<U>, value_type>::value,
                       int> = 0>
  constexpr explicit strong_value(U&& u) noexcept : value{std::forward<U>(u)} {}

  constexpr strong_value() noexcept : value{} {}

  value_type value;
};

template <class Type, class Tag, class... Params>
struct number : derive_t<number<Type, Tag, Params...>,
                         arithmetic,
                         comparable,
                         arithmetically_compatible_with<Type, cast_to_then_construct_t<Type, black_magic::deduce>>,
                         comparable_with<Type>,
                         Params...> {
  using value_type = Type;

  static_assert(std::is_arithmetic<value_type>::value,
                "number expects a literal as base (first template parameter)");

  constexpr number() noexcept = default;

  template <class U,
            std::enable_if_t<
                std::is_constructible<value_type, std::decay_t<U>>::value,
                int> = 0>
  constexpr explicit number(U&& u) noexcept : value{std::forward<U>(u)} {}

  value_type value;
};

}  // namespace strong_types
}  // namespace dpsg

#endif  // GUARD_DPSG_STRONG_TYPES:_HPP

// Allows streaming to/from std::basic_[io]stream
namespace st = dpsg::strong_types;
namespace meta = st::black_magic;
struct streamable {
  template <class T>
  struct type
      : meta::for_each<
            meta::tuple<meta::tuple<st::shift_left, std::basic_istream<char>>,
                        meta::tuple<st::shift_right, std::basic_ostream<char>>>,
            meta::apply<st::implement_binary_operation, T>> {};
};

// Constants
constexpr int MAX_X = 17630;
constexpr int MAX_Y = 9000;
constexpr int DANGER_ZONE = 5000;
constexpr int WIND_RANGE = 1280;
constexpr int SHIELD_RANGE = 2200;
constexpr int CONTROL_RANGE = 2200;
constexpr int SHIELD_DURATION = 12;
constexpr int INFINITE_DISTANCE = numeric_limits<int>::max();
constexpr int ATTACK_RANGE = 800;
constexpr int HERO_SIGHT_RANGE = 2200;
constexpr int BASE_SIGHT_RANGE = 6000;

constexpr auto always = [](auto&& in) { return [in = forward<decltype(in)>(in)] (auto&&...) { return in; }; };

// Type definitions

struct entity_id_t
    : st::strong_value<int, entity_id_t, st::comparable,
                       streamable> {};

struct position_t {
    int x; 
    int y;

    friend constexpr position_t operator+(position_t left, position_t right) noexcept {
        return position_t{.x = left.x + right.x, .y = left.y + right.y};
    }

    friend constexpr position_t operator-(position_t left, position_t right) noexcept {
        return position_t{.x = left.x - right.x, .y = left.y - right.y};
    }

    friend constexpr position_t abs(position_t in) noexcept {
        return position_t{.x = abs(in.x), .y = abs(in.y)};
    }
};

constexpr float distance(position_t left, position_t right) {
  int dx = left.x - right.x;
  int dy = left.y - right.y;
  return sqrt(dx * dx + dy * dy);
}

position_t travel(position_t start, position_t end, int dist) {
  float dx = end.x - start.x;
  float dy = end.y - start.y;
  float dh = distance(start, end);
  float cos = dx / dh;
  float sin = dy / dh;
  return position_t{.x = static_cast<int>(cos * dist + start.x),
                    .y = static_cast<int>(sin * dist + start.y)};
}

namespace detail {

template<class T, class = void>
struct has_position_t: false_type {};
template<class T>
struct has_position_t<T, void_t<decltype(declval<T>().position())>> : true_type {};
}

struct get_position_t {
  template <class T, enable_if_t<detail::has_position_t<T>::value, int> = 0>
  constexpr position_t operator()(T &&e) const noexcept {
    return e.position();
  }

  template <class T>
  constexpr position_t operator()(const reference_wrapper<T> &ref) {
    return get_position_t{}(ref.get());
  }

  constexpr const position_t &operator()(const position_t &p) const noexcept {
    return p;
  }

  constexpr position_t operator()(position_t &&p) const noexcept { return p; }
} constexpr get_position;

template <class T> struct closest_to {
  template <class U> constexpr explicit closest_to(U &&u) noexcept : target{forward<U>(u)} {}

  constexpr closest_to(const closest_to<T>&) noexcept = default;
  constexpr closest_to(closest_to<T>&&) noexcept = default;
  
  template<class V, class U>
  constexpr bool operator()(const U &left,
                            const V &right) const noexcept {
    return distance(get_position(left), get_position(target)) <
           distance(get_position(right), get_position(target));
  }

  private:
  T target;
};
template<class T>
closest_to(T&&) -> closest_to<decay_t<T>>;

using health = st::number<int, struct health_tag, streamable>;
using mana = st::number<int, struct mana_tag, streamable>;
using hero_id = st::number<int, struct hero_id_tag>;
enum class player { me = 0, opponent = 1 };

class entity {
public:
  enum type {
    monster = 0,
    hero = 1,
    opponent = 2,
  };

private:
  entity_id_t id_;    // Unique identifier
  int type_;          // 0=monster, 1=your hero, 2=opponent hero
  position_t pos_;    // Position of this entity
  int shield_life_;   // Ignore for this league; Count down until shield spell
                      // fades
  int is_controlled_; // Ignore for this league; Equals 1 when this entity is
                      // under a control spell
  int health_;        // Remaining health of this monster
  int vx_;            // Trajectory of this monster
  int vy_;
  int near_base_;  // 0=monster with no target yet, 1=monster targeting a base
  int threat_for_; // Given this monster's trajectory, is it a threat to 1=your
                   // base, 2=your opponent's base, 0=neither

public:
  constexpr bool controlled() const noexcept { return is_controlled_ == 1; }
  constexpr bool is_monster() const noexcept { return type_ == type::monster; }
  constexpr bool is_hero() const noexcept { return type_ == type::hero; }
  constexpr bool is_opponent() const noexcept {
    return type_ == type::opponent;
  }
  constexpr bool shielded() const noexcept { return shield_life_ > 0; }
  constexpr bool targeting() const noexcept { return near_base_ == 1; }
  constexpr bool threat_for(player p) const noexcept {
    return (p == player::me && threat_for_ == 1) ||
           (p == player::opponent && threat_for_ == 2);
  }

  constexpr const position_t &position() const noexcept { return pos_; }
  constexpr const entity_id_t &id() const noexcept { return id_; }
  constexpr int health() const noexcept { return health_; }

  friend istream &operator>>(istream &, entity &);
};

using entity_list = std::vector<entity>;

struct wait_t {} constexpr wait;
struct walk {
    constexpr explicit walk(position_t pos) : position{pos} {}

    position_t position;
};
struct shield {
  entity_id_t id;

  constexpr explicit shield(entity_id_t id) : id{id} {}
};
struct control {
  constexpr explicit control(entity_id_t id, position_t position)
      : id{id}, position{position} {}

  entity_id_t id;
  position_t position;
};
struct wind {
    constexpr explicit wind(position_t position) : position{position} {}

    position_t position;
};

template<class ...Fs>
struct overload_set : Fs... {
    constexpr overload_set() = delete;
    template<class ...Gs, std::enable_if_t<sizeof...(Fs) == sizeof...(Gs), int> = 0>
    constexpr overload_set(Gs&&... gs) : Fs{std::forward<Gs>(gs)}... {}

    using Fs::operator()...;
};
template<class ...Fs> 
overload_set(Fs&&...) -> overload_set<std::remove_cv_t<std::remove_reference_t<Fs>>...>;

class decision {
    std::variant<wait_t, walk, shield, control, wind> _value;

 public:
    constexpr decision(wait_t w) : _value{w} {}
    constexpr decision(walk m) : _value{m} {}
    constexpr decision(wind w) : _value{w} {}
    constexpr decision(shield s) : _value{s} {}
    constexpr decision(control c) : _value{c} {}

    template<class Wait, class Walk, class Wind, class Shield, class Control> 
    constexpr decltype(auto) visit(Wait&& waitf, Walk&& walkf, Wind&& windf, Shield&& shieldf, Control&& controlf) {
      return std::visit(
          overload_set{forward<Walk>(walkf), forward<Wait>(waitf),
                         forward<Wind>(windf), forward<Shield>(shieldf),
                         forward<Control>(controlf)},
          _value);
    }
};

using entity_ref = reference_wrapper<const entity>;
using entity_ref_opt = optional<entity_ref>;

////////////////////////////////////////
// GAME STATE 
////////////////////////////////////////

namespace detail {
    template<class T, class C>
    struct contains : false_type {};
    template<class T, template<class...> class C, class ...Args>
    struct contains<T, C<Args...>> : bool_constant<disjunction_v<is_same<T, Args>...>> {};
    struct get_entity_id_t {
        template<class ...Args>
        entity_id_t operator()(entity_id_t id, Args&&...) const {
            return id;
        }
        template<class T, class ...Args, enable_if_t<!is_same_v<decay_t<T>, entity_id_t>, int> = 0>
        entity_id_t operator()(T&&, Args&&... args) const {
            return operator()(forward<Args>(args)...);
        }
    } constexpr get_entity_id;
    template<class ...Ts>
    void reset_cache(tuple<Ts...>& tpl) {
        (get<Ts>(tpl).targets_.clear(), ...);
    }
}
/// \brief true_type if type T is contained within template container C
template<class T, class C>
constexpr static inline bool contains_v = detail::contains<T, C>::value;

/** \brief cache values common to the whole game */
struct general_cache {
    template<class S>
    struct target_cache {
        set<entity_id_t> targets_;

        bool target(entity_id_t id) {
            return targets_.insert(id).second;
        }

        bool is_targeted(entity_id_t id) {
            return targets_.count(id) > 0;
        }
    };
    template <class... Ss>
    using spell_tuple_f = tuple<target_cache<Ss>...>;
    using spell_tuple = spell_tuple_f<shield, control>;

    entity_ref_opt closest_from_opponent_base;
    entity_ref_opt opponent_runner;
    spell_tuple spell_target_cache;

    /// \returns true if target with given id is not under the influence of 
    /// the same spell
    template<class T, class... Args>
    bool target(Args&&... args) {
        if constexpr (contains_v<T, spell_tuple>) {
            return get<T>(spell_target_cache).target(detail::get_entity_id(args...));
        }
        else {
            return true;
        }
    }

    /// \brief resets relevant parts of the cache
    void reset() {
        closest_from_opponent_base = nullopt;
        opponent_runner = nullopt;
        detail::reset_cache(spell_target_cache);
    }
};

/** \brief cache values relative to a given hero */
struct hero_cache {
    hero_cache(hero_id hero) : hero_{hero}, ref{} {}

    void reset() { ref=nullptr; }
    void set_ref(entity& e) { ref = addressof(e); }
    position_t target;

private:
    hero_id hero_;
    entity* ref;
};

/** \brief contains everything related to the state of the game
*/
struct game_state {
private:
  health health_pool[2];
  mana mana_pool[2];
  std::vector<entity> heroes_{};
  std::vector<entity> monsters_{};
  std::vector<entity> opponents_{};

  general_cache cache_{};
  std::vector<hero_cache> hero_cache_;


public:
  game_state(position_t base, int hero_nb)
      : base{base}, opponent_base{position_t{.x = MAX_X, .y = MAX_Y} - base} {
          for (hero_id i{0}; i < hero_nb; ++i)
          hero_cache_.emplace_back(i);
      }
  const std::vector<entity> &monsters{monsters_};
  const std::vector<entity> &heroes{heroes_};
  const std::vector<entity> &opponents{opponents_};
  const position_t base;
  const position_t opponent_base;
  constexpr mana player_mana(player p) const {
    return mana_pool[static_cast<int>(p)];
  }

  optional<entity> closest_from_base(player p = player::me) {
      if (monsters.empty()) {
          return {};
      }
      if (p == player::me) {
          return monsters.front();
      }
      if (!cache_.closest_from_opponent_base.has_value()) {
        cache_.closest_from_opponent_base = ref(*min_element(
            monsters.begin(), monsters.end(), closest_to{opponent_base}));
      }
      return *cache_.closest_from_opponent_base;
  }

  void reset() {
    monsters_.clear(); 
    heroes_.clear(); 
    opponents_.clear(); 
    for(auto& cache : hero_cache_) {
        cache.reset();
    }
    cache_.reset();
  }

  hero_cache& cache_for(hero_id id) {
      return hero_cache_[id.value];
  }

  position_t target_of(hero_id id) {
      return cache_for(id).target;
  }

  constexpr mana my_mana() const { return player_mana(player::me); }

  constexpr mana opponent_mana() const { return player_mana(player::opponent); }

  constexpr health player_health(player p) const {
    return health_pool[static_cast<int>(p)];
  }

  constexpr health my_health() const { return player_health(player::me); }

  constexpr health opponent_health() const {
    return player_health(player::opponent);
  }

  entity hero(hero_id id) const noexcept {
      return heroes[id.value];
  }

  template <class Spell, class... Args,
            enable_if_t<is_constructible_v<Spell, Args...>, int> = 0>
  optional<Spell> reserve_mana(Args &&...args) noexcept {
    if (my_mana() >= 10) {
      bool can_target = cache_.target<Spell>(args...);
      if (!can_target) {
          return {};
      }
      mana_pool[static_cast<int>(player::me)] -= 10;
      return Spell{forward<Args>(args)...};
    }
    return {};
  }

  template<class Spell, class F, enable_if_t<!is_constructible_v<Spell, F>, int> = 0>
  optional<Spell> reserve_mana(F&& construct_from_params) noexcept {
    return construct_from_params([this](auto &&...params) {
      return reserve_mana<Spell>(std::forward<decltype(params)>(params)...);
    });
  }

  void add_hero(const entity& entity) {
      using h = struct hero;
      heroes_.push_back(entity);
      hero_cache_[heroes_.size() - 1].set_ref(heroes_.back());
  }

  optional<position_t> opponent_runner() {
    if (cache_.opponent_runner.has_value())
      return cache_.opponent_runner->get().position();

    for (auto& op : opponents_) {
        if (closest_to{op}(base, opponent_base)) {
            cache_.opponent_runner = op;
            return op.position();
        }
    }
    return {};
  }

  friend std::istream &operator>>(std::istream &in, game_state &state);
};
using decision_function = std::function<decision(hero_id, game_state &)>;

struct within_t {
  template <class T, class U>
  constexpr bool operator()(const T &left, const U &right,
                            int dist) const noexcept {
    return distance(get_position(left), get_position(right)) <= dist;
  }
} constexpr within;

// IO

std::istream& operator>>(std::istream& in, position_t& position) {
    return in >> position.x >> position.y;
}

std::istream& operator>>(std::istream& in, entity& m) {
  return in >> m.id_ >> m.type_ >> m.pos_ >> m.shield_life_ >> m.is_controlled_ >> m.health_ >> m.vx_ >>
      m.vy_ >> m.near_base_ >> m.threat_for_;
}

std::istream& operator>>(std::istream& in, game_state& state) {
  for (int i = 0; i < 2; i++) {
    in >> state.health_pool[i] >> state.mana_pool[i];
  }
  int entity_count; // Amount of heros and monsters you can see
  in >> entity_count;

  state.reset();
  for (int i = 0; i < entity_count; i++) {
    entity entit;
    in >> entit;
    if (entit.is_monster()) {
      state.monsters_.push_back(entit);
    } else if (entit.is_hero()) {
      state.add_hero(entit);
    } else {
      state.opponents_.push_back(entit);
    }
  }

  sort(state.monsters_.begin(), state.monsters_.end(), closest_to{state.base});

  return in;
}

std::ostream& operator<<(std::ostream& out, const position_t& position) {
    return out << position.x << " " << position.y;
}

std::ostream &operator<<(std::ostream &out, decision d) {
  using R = std::ostream &;
  return d.visit(
      [&out](wait_t _) -> R { return out << "WAIT"; },
      [&out](walk w) -> R { return out << "MOVE " << w.position; },
      [&out](wind w) -> R { return out << "SPELL WIND " << w.position; },
      [&out](shield w) -> R { return out << "SPELL SHIELD " << w.id; },
      [&out](control w) -> R {
        return out << "SPELL CONTROL " << w.id << " " << w.position;
      });
}

// algorithms

template<class T, class F>
bool contains(const T& t, const F& f) {
    using std::begin, std::end;
    return find_if(begin(t), end(t), f) != end(t);
}

// Decision tree management

namespace detail {
template <int I, class R, class... Ps, class... Args>
constexpr R expand_decision_tree(const tuple<Args...> &tpl,
                                    Ps &&...ps) noexcept {
  if constexpr (I == sizeof...(Args)) {
    static_assert(I == -1, "Dont let this happen, return a non optional value");
  } else {
    using ret_t = decltype(get<I>(tpl)(forward<Ps>(ps)...));
    if constexpr (dpsg::is_template_instance_v<ret_t, optional>) {
      const auto v = get<I>(tpl)(forward<Ps>(ps)...);
      if (v.has_value()) {
        return v.value();
      }
      return expand_decision_tree<I + 1, R>(tpl, forward<Ps>(ps)...);
    } else {
      const auto v = get<I>(tpl)(forward<Ps>(ps)...);
      return v;
    }
  }
}

template <class R, class... Args, class... Ps>
constexpr R expand_decision_tree(const tuple<Args...> &tpl,
                                 Ps &&...ps) noexcept {
  return expand_decision_tree<0, R>(tpl, forward<Ps>(ps)...);
}
} // namespace detail

template <class... Decisions> struct decide {
  template <class... Args>
  constexpr decide(Args &&...args) : decision_trees{forward<Args>(args)...} {}

  constexpr decision operator()(hero_id id, game_state &state) {
    return detail::expand_decision_tree<decision>(decision_trees, id, state);
  }

private:
  tuple<Decisions...> decision_trees;
};
template<class ...Args>
decide(Args&&...) -> decide<decay_t<Args>...>;

// decision utilities
optional<position_t> target_for_wind(const entity &hero,
                                     game_state& state) {

  struct within_attack_distance {
      entity target;
      int range = ATTACK_RANGE;
    constexpr bool operator()(const entity &other) const noexcept {
      return within(target, other,  range);
    }
  };

  for (const auto &monster : state.monsters) {
    int dist_from_base = distance(monster.position(), state.base);
    bool endengering_base = dist_from_base <= DANGER_ZONE;
    bool in_range = within(hero, monster, WIND_RANGE);

    if (!in_range) continue;

    if (endengering_base && !monster.shielded()) {
      return monster.position();
    }
    if (dist_from_base > DANGER_ZONE + WIND_RANGE) {
        return {};
    }
    for (const auto& opponent : state.opponents) {
        if (within(opponent, monster, WIND_RANGE)) {
            return monster.position();
        }
    }
  }
  return {};
}

optional<entity> closest_threat(position_t base, const entity_list &monsters,
                                int cutoff = INFINITE_DISTANCE) {
  for (const auto &monster : monsters) {
    auto dist_to_base = distance(monster.position(), base);
    if (dist_to_base > cutoff) {
      return {};
    }
    if (monster.threat_for(player::me)) {
      return monster;
    }
  }
  return {};
}

optional<entity> closest_threatening_monster(const entity_list &monsters,
                                             const entity &hero) {
  auto it = min_element(monsters.begin(), monsters.end(),
                        [&hero](const entity &left, const entity &right) {
                          if (left.threat_for(player::opponent) &&
                              !right.threat_for(player::opponent)) {
                            return false;
                          }
                          if (!left.threat_for(player::opponent) &&
                              right.threat_for(player::opponent)) {
                            return true;
                          }
                          return distance(hero.position(), left.position()) <
                                 distance(hero.position(), right.position());
                        });
  if (it != monsters.end()) {
    return *it;
  }
  return {};
}

optional<entity> best_controllable_monster(const entity &hero, game_state& state) {
  entity_list controllable_monsters;
  controllable_monsters.reserve(state.monsters.size());

  struct controllable_by {
    entity hero;
    const entity_list &heroes;
    const entity_list &opponents;
    position_t base;

    bool operator()(const entity &monster) const noexcept {
      for (int i = 0; i < max(heroes.size(), opponents.size()); ++i) {
        if ((i < heroes.size() &&
             distance(heroes[i].position(), monster.position()) <
                 ATTACK_RANGE) ||
            (i < opponents.size() &&
             distance(opponents[i].position(), monster.position()) <
                 ATTACK_RANGE * 1.5)) {
          return false;
        }
        if (distance(monster.position(), base) < DANGER_ZONE)  {
          return false;
        }
      }
      return !monster.shielded() && !monster.controlled() &&
             !monster.threat_for(player::opponent) && monster.health() > 16 &&
             distance(hero.position(), monster.position()) < CONTROL_RANGE;
    }
  };

  copy_if(state.monsters.begin(), state.monsters.end(),
          back_inserter(controllable_monsters), controllable_by{hero, state.heroes, state.opponents, state.base});

  auto it =
      min_element(controllable_monsters.begin(), controllable_monsters.end(),
                  [&](const entity &left, const entity &right) {
                    if (left.health() != right.health()) {
                      return left.health() > right.health();
                    }
                    return !closest_to{hero.position()}(left, right);
                  });
  if (it != controllable_monsters.end()) {
    return *it;
  }
  return {};
}

optional<entity> shield_target(const entity& hero, const entity_list& monsters, position_t opponent_base) {
    int best_distance = INFINITE_DISTANCE;
    const entity* result = nullptr;
    for (const auto& monster : monsters) {
        if (!monster.threat_for(player::opponent) || distance(hero.position(), monster.position()) > SHIELD_RANGE) {
            continue;
        }
        int dist_from_ob = distance(monster.position(), opponent_base);
        if (dist_from_ob < best_distance && 
            dist_from_ob < 12 * 400 + 300) { // 12 turn shield * 400 monster mvt + 300 dmg range
          result = &monster;
          best_distance = dist_from_ob;
        }
    }
    return result == nullptr ? nullopt : optional{*result};
}

/********************************/
/*  hero decisions              */ 
/********************************/

optional<decision> wind_away_if_needed(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);

  if (auto target_monster = target_for_wind(this_hero, state);
      target_monster.has_value()) {
    if (auto spell = state.reserve_mana<wind>(state.opponent_base); spell.has_value()) {
      return spell.value(); 
    }
  }
  return {};
}

optional<decision> shadow_opponent_runner(hero_id id, game_state& state) {
  const auto threat = closest_threat(state.base, state.monsters);
  const auto this_hero = state.hero(id);
  const position_t target = state.target_of(id);
  const auto opponent_runner = find_if(
      state.opponents.begin(), state.opponents.end(), [&](const entity &ent) {
        return within(ent, state.base, DANGER_ZONE * 1.1);
      });

  if (opponent_runner != state.opponents.end()) {
    if (const auto spell = state.reserve_mana<shield>(this_hero.id());
        spell.has_value() && state.opponent_mana() >= 10 &&
        !this_hero.shielded() && threat.has_value()) {
      return spell.value();
    }
    if (state.my_mana() >= 30 && !opponent_runner->shielded() &&
        !opponent_runner->controlled() &&
        distance(opponent_runner->position(), this_hero.position()) <
            CONTROL_RANGE) {
        if (auto spell = state.reserve_mana<control>(opponent_runner->id(), state.opponent_base))
      return spell;
    }
  }
  return {};
}

optional<decision> attack_closest_threat_from_base(hero_id, game_state& state) {
  const auto threat = closest_threat(state.base, state.monsters);
  if (threat.has_value()) {
    return walk{threat->position()};
  }
  return {};
}

template<class T>
struct set_target {
    template<class U>
    constexpr set_target(U&& f) : compute_target(forward<U>(f)) {}
    T compute_target;
    optional<decision> operator()(hero_id id, game_state& state) const {
        state.cache_for(id).target = compute_target(id, state);
        return {};
    }
};
template<class T>
set_target(T&&) -> set_target<decay_t<T>>;

decision go_to_target(hero_id id, game_state& state) {
    return walk{state.target_of(id)};
}

constexpr auto away_from_base = [](int dist) {
  return [=](hero_id, game_state &state) {
    constexpr int s = min(MAX_X, MAX_Y);
    const position_t t = abs(state.base - position_t{s, s});
    return travel(state.base, t, dist);
  };
};
constexpr auto middle_of_the_map = [] (hero_id, game_state& state) {
    auto middle = position_t{
        .x = abs(state.base.x - state.opponent_base.x) / 2,
        .y = abs(state.base.y - state.opponent_base.y) / 2,
    };
    return middle;
};

optional<decision> shield_opponent_monster(hero_id id, game_state &state) {
  const auto this_hero = state.hero(id);
  if (state.my_mana() > 50) {
    if (auto monster =
            shield_target(this_hero, state.monsters, state.opponent_base);
        monster.has_value()) {
      if (auto spell = state.reserve_mana<shield>(monster->id());
          spell.has_value())
        return spell.value();
    }
  }
  return {};
}

optional<decision> control_monster_to_attack(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);
  if (state.my_mana() > 80) {
    if (auto closest = best_controllable_monster(this_hero, state);
        closest.has_value()) {
      if (auto spell =
              state.reserve_mana<control>(closest->id(), state.opponent_base))
        return spell;
    }
  }
  return {};
}

optional<decision> dont_attack_threats_to_opponent(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);
  auto closest = closest_threatening_monster(state.monsters, this_hero);

  if (!closest.has_value() || closest->threat_for(player::opponent)) {
    return walk{state.target_of(id)};
  }
  return {};
}

optional<decision> attack_closest_threat_from_self(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);
  auto closest = closest_threatening_monster(state.monsters, this_hero);

  if (closest.has_value())
    return walk{closest->position()};
  return {};
}

optional<decision> last_resort_control(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);
  const auto threat = closest_threat(state.base, state.monsters, DANGER_ZONE * 1.2);

  if (threat.has_value() && closest_to(state.base)(threat.value(), this_hero) &&
      !within(this_hero, threat.value(), WIND_RANGE) && within(this_hero, threat.value(), CONTROL_RANGE) &&
      within(threat.value(), state.base, WIND_RANGE + 800)) {
    if (auto spell =
            state.reserve_mana<control>(threat->id(), state.opponent_base)) {
      return spell.value();
    }
  }
  return {};
}

optional<decision> shield_self(hero_id id, game_state& state) {
  const auto this_hero = state.hero(id);
  const auto opponent_runner = state.opponent_runner();
  const auto closest_monster = state.closest_from_base();

  if (auto spell = state.reserve_mana<shield>([&](auto shield)
                                                  -> optional<struct shield> {
        if (opponent_runner.has_value() && closest_monster.has_value() &&
            state.opponent_mana() >= 10 && !this_hero.shielded() &&
            within(closest_monster.value(), state.base, DANGER_ZONE)) {
          return shield(this_hero.id());
        }
        return {};
      });
      spell.has_value()) {
    return spell.value();
  }
  return {};
}

optional<decision> go_to_opponent_runner(hero_id id, game_state &state) {
  const auto opponent_runner = state.opponent_runner();
  if (opponent_runner.has_value()) {
    if (state.my_mana() > 10)
      return walk{opponent_runner.value()};
  }
  return {};
}

struct attack_around_target {
  int cutoff;

  optional<decision> operator()(hero_id id, game_state &state) const {
    const auto target = state.target_of(id);
    if (auto closest = closest_threat(target, state.monsters);
        closest.has_value() && within(*closest, target, cutoff) &&
        closest->threat_for(player::me)) {
      return walk{closest->position()};
    }
    return {};
  }
};

constexpr auto home_defender = decide{
    set_target(away_from_base(DANGER_ZONE * 1.1)), wind_away_if_needed,
    shadow_opponent_runner, attack_closest_threat_from_base, go_to_target};

constexpr auto point_runner = decide{set_target(middle_of_the_map),
                                     wind_away_if_needed,
                                     shield_opponent_monster,
                                     control_monster_to_attack,
                                     dont_attack_threats_to_opponent,
                                     attack_closest_threat_from_self,
                                     go_to_target};

constexpr auto mage_defender = decide{set_target(away_from_base(DANGER_ZONE)),
                                      last_resort_control,
                                      wind_away_if_needed,
                                      shield_self,
                                      attack_closest_threat_from_base,
                                      go_to_opponent_runner,
                                      attack_around_target{2000},
                                      go_to_target};



int main() {
  position_t base;
  cin >> base;
  cin.ignore();
  int heroes_per_player; // Always 3
  cin >> heroes_per_player;
  game_state state{base, heroes_per_player};
  cin.ignore();
  std::vector<decision_function> heroes_actions;
  heroes_actions.resize(heroes_per_player);
  heroes_actions[0] = mage_defender;
  heroes_actions[1] = home_defender;
  heroes_actions[2] = point_runner;

  // game loop
  while (1) {
    cin >> state;

    hero_id current_hero{0};
    for (const auto &act : heroes_actions) {
      cout << act(current_hero, state) << std::endl;
      current_hero++;
    }
  }
}
