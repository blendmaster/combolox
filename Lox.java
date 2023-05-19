import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.BiFunction;
import java.util.List;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.Arrays;
import java.util.Optional;

class Lox {
  public static void main(String[] args) {
    System.out.println("hi");

  }
}

// since java.util.function stops at BiFunction.
interface F3<A, B, C, T> { T apply(A a, B b, C c); }
interface F4<A, B, C, D, T> { T apply(A a, B b, C c, D d); }

// parsing expression grammar free applicative,
// parses a stream of Ps into a T
sealed interface Peg<P, T> permits Peg.Success, Peg.Fail, Peg.Terminal, Peg.Sequence, Peg.Choice, Peg.OneOrMore, Peg.Ref {
  // matches a P by predicate, mapped into the output type.
  record Terminal<P, T>(Predicate<P> predicate, Function<P, T> map) implements Peg<P,T> {}

  // succeeds but parsing nothing, aka pure
  record Success<P, T>(T value) implements Peg<P, T> {}

  // always fails
  record Fail<P, T>(String errorMessage) implements Peg<P,T> {}

  // parses first and then next, aka <*> in haskell
  record Sequence<P, A, T>(Peg<P, A> first, Peg<P, Function<A, T>> next) implements Peg<P,T> {}
  // parses left, then if it fails, parses right.
  record Choice<P, T>(Peg<P,T> left, Peg<P,T> right) implements Peg<P,T> {}

  // Parse one or more of the same, collected into a final T using the stream Collector interface.
  // which is close enough to Haskell Foldable and has useful helpers like Collectors.toList().
  // Note this requires at matching one input, i.e. the collection will be non-empty. This is
  // because you can recover zero-or-more from Choice(OneOrMore(), Success(empty)), but
  // you can't recover one-or-more from Sequence(first, ZeroOrMore()). you can't get back the
  // Accumulator type from the final T to prepend the first.
  record OneOrMore<P, I, A, T>(Peg<P, I> peg, Collector<I, A, T> collector) implements Peg<P,T> {}

  // another way to do OneOrMore is
  //
  // record Continue<P, A, T>(Peg<A> first, Function<A, Peg<P,T>> cont) implements Peg<P,T> {}
  //
  // where each time the parse succeeds, run it through `cont` to get the next parse
  // to try. You can build a list up this way. 
  // However, the arbitrary logic you could put inside Cont (i.e. a free monad) is
  // more powerful than we need to just do OneOrMore.
  
  // kludge for circular references in grammars; create a Ref placeholder, then later
  // set the ref with its final value (that can refer to other pegs), with the mapping
  // composed together for when it's needed.
  record Ref<P, A, T>(AtomicReference<Peg<P, A>> peg, Function<A, T> map) implements Peg<P,T> {
    void set(Peg<P, A> value) { peg.set(value); }
  }

  // deriving Functor
  // XXX java type system not powerful enough for some of the type parameters in here.
  @SuppressWarnings("unchecked")
  default <A, R, I> Peg<P, R> map(Function<T, R> f) {
    return switch (this) {
        case Terminal(var predicate, var fmap) -> new Terminal<P, R>(
            predicate, fmap.andThen(f));
        case Success(var value) -> new Success<>(f.apply(value));
        case Fail(var message) -> new Fail<>(message);
        case Sequence(var first, var next) ->
          new Sequence<P, A, R>(
              ((Peg<P, A>)first), ((Peg<P, Function<A, T>>)next).map(f::compose));
        case Choice(var left, var right) -> new Choice<>(left.map(f), right.map(f));
        case OneOrMore(var peg, var collector) -> 
          new OneOrMore<P, I, A, R>(
              ((Peg<P, I>)peg), Collectors.collectingAndThen(
                ((Collector<I, A, T>)collector), f));
        // the peg AtomicReference is shared (and eventually set)
        case Ref(var peg, var map) -> new Ref(peg, map.andThen(f));
    }; 
  }

  // helpers

  // convenience for when you just need a marker for whatever got parsed.
  default <R> Peg<P, R> as(R r) {
    return map(__ -> r);
  }

  static <P, T> Ref<P, T, T> ref() {
    return new Ref<>(new AtomicReference<>(), Function.identity());
  }

  // fold multiple choices into a linked list, essentially.
  @SafeVarargs
  static <P, T> Peg<P,T> choice(Peg<P,T> choice1, Peg<P,T> choice2, Peg<P,T>... more) {
    return Arrays.stream(more)
      .reduce(new Choice<>(choice1, choice2), (cs, c) -> new Choice<>(cs, c));
  }

  // aka "lift", so you don't have to think about currying as much.
  static <P, A, B, R> Peg<P, R> sequence(BiFunction<A, B, R> f, Peg<P, A> pegA, Peg<P, B> pegB) {
    return new Sequence<>(pegA, pegB.map(b -> a -> f.apply(a, b)));
  }

  static <P, A, B, C, R> Peg<P, R> sequence(F3<A, B, C, R> f, Peg<P, A> pegA, Peg<P, B> pegB, Peg<P, C> pegC) {
    return new Sequence<>(pegA, 
        new Sequence<>(pegB,
          pegC.map(c -> b -> a -> f.apply(a, b, c))));
  }

  // parse actual in the middle, surrounded by stuff you don't need the value of
  static <P, L, R, T> Peg<P, T> surround(Peg<P, L> left, Peg<P,T> actual, Peg<P, R> right) {
    return sequence((_1, a, _2) -> a, left, actual, right);
  }

  static <P, T> Peg<P, Optional<T>> optional(Peg<P, T> peg) {
    return choice(peg.map(Optional::of), new Success<>(Optional.empty()));
  }

  static Peg<Character, String> repeatString(Peg<Character, String> peg) {
    return new OneOrMore<>(peg.map(a -> (CharSequence)a), Collectors.joining());
  }

  static <P, I, A, T> Peg<P, T> zeroOrMore(Peg<P, I> peg, Collector<I, A, T> collector) {
    return choice(
        new OneOrMore<>(peg, collector),
        // kind of unfortunate we need to strictly create the empty case here, oh well
        new Success<>(collector.finisher().apply(collector.supplier().get())));
  }

  static <P> Peg<P, P> matching(Predicate<P> predicate) {
    return new Terminal<>(predicate, Function.identity());
  }

  static <P> Peg<P, P> literal(P value) {
    return matching(value::equals);
  }

  // extremely efficient string literal parser, character by character.
  static Peg<Character, String> string(String value) {
    if (value.length() == 1) {
      return new Terminal<>(((Character)value.charAt(0))::equals, String::valueOf);
    }

    // fold characters into a sequence of literals that foldss results back into a string.
    return value.chars()
      .mapToObj(c -> literal(Character.valueOf((char)c)).map(String::valueOf))
      .reduce(
          new Success<>(""), 
          (prev, next) -> new Sequence<>(prev, next.map(n -> p -> p.concat(n))));
  }
}

// Lox AST
/*
 expression     → literal
               | unary
               | binary
               | grouping ;

literal        → NUMBER | STRING | "true" | "false" | "nil" ;
grouping       → "(" expression ")" ;
unary          → ( "-" | "!" ) expression ;
binary         → expression operator expression ;
operator       → "==" | "!=" | "<" | "<=" | ">" | ">="
               | "+"  | "-"  | "*" | "/" ;

could you write a Peg for the above language as Peg<Peg<?>>? yes, I think.
*/

class LoxAst {

  sealed interface Expression permits Literal, Unary, Binary, Grouping {}

  enum Operator {
    EQUALS, NEQUALS, LESS, LESSEQUAL, GREATER, GREATEREQUAL, PLUS, MINUS, MULT, DIV
  }

  record Binary(Expression left, Operator operator, Expression right) implements Expression {}

  enum UnaryOp {
    NEGATIVE, NOT
  }
  record Unary(UnaryOp op, Expression expression) implements Expression {}

  record Grouping(Expression expression) implements Expression {}

  sealed interface Literal extends Expression permits LoxNumber, LoxString, LoxTrue, LoxFalse, LoxNil {}
  record LoxNumber(float value) implements Literal {}
  record LoxString(String value) implements Literal {}
  record LoxTrue() implements Literal {}
  record LoxFalse() implements Literal {}
  record LoxNil() implements Literal {}

  // Parser

  static Peg.Ref<Character, Expression, Expression> ExpressionPeg = Peg.ref();

  static Peg<Character, String> digit = 
    Peg.<Character>matching(Character::isDigit).map(String::valueOf);

  static Peg<Character, String> digits = Peg.repeatString(digit);

  static Peg<Character, Float> number =
    Peg.sequence(
        (whole, decimal) -> Float.parseFloat(whole + decimal.orElse("")),
        digits,
        Peg.optional(
          Peg.sequence(String::concat, Peg.string("."), digits)));

  static Peg<Character, Literal> LiteralPeg = Peg.choice(
      Peg.string("nil").as(new LoxNil()),
      Peg.string("true").as(new LoxTrue()),
      Peg.string("false").as(new LoxFalse()),
      number.map(LoxNumber::new)
      // TODO alpha
    );

  static Peg<Character, Unary> UnaryPeg = Peg.sequence(
      Unary::new,
      Peg.choice(
        Peg.string("-").as(UnaryOp.NEGATIVE),
        Peg.string("!").as(UnaryOp.NOT)),
      ExpressionPeg);

  static Peg<Character, Operator> OperatorPeg = Peg.choice(
      Peg.string("==").as(Operator.EQUALS),
      Peg.string("!=").as(Operator.NEQUALS),
      Peg.string("<") .as(Operator.LESS),
      Peg.string("<=").as(Operator.LESSEQUAL),
      Peg.string(">") .as(Operator.GREATER),
      Peg.string(">=").as(Operator.GREATEREQUAL),
      Peg.string("+") .as(Operator.PLUS),
      Peg.string("-") .as(Operator.MINUS),
      Peg.string("*") .as(Operator.MULT),
      Peg.string("/") .as(Operator.DIV));

  static Peg<Character, Binary> BinaryPeg = Peg.sequence(
      Binary::new,
      ExpressionPeg,
      OperatorPeg,
      ExpressionPeg);

  static Peg<Character, Grouping> GroupingPeg = 
    Peg.surround(Peg.string("("), ExpressionPeg, Peg.string(")")).map(Grouping::new);

  {
    // XXX requires upcast for some reason.
    ExpressionPeg.set(Peg.<Character, Expression>choice(
          LiteralPeg.map(a -> (Expression)a),
          UnaryPeg.map(a -> (Expression)a),
          BinaryPeg.map(a -> (Expression)a),
          GroupingPeg.map(a -> (Expression)a)));
  }
}
