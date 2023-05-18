import java.util.function.Function;
import java.util.function.BiFunction;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.Arrays;

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
  // matches a single value
  record Terminal<P, T>(P value, Function<P, T> map) implements Peg<P,T> {}

  // succeeds parsing nothing, aka pure
  record Success<P, T>(T value) implements Peg<P, T> {}
  // always fails
  record Fail<P, T>(String errorMessage) implements Peg<P,T> {}

  // parses first and then next, aka <*> in haskell
  record Sequence<P, A, T>(Peg<P, A> first, Peg<P, Function<A, T>> next) implements Peg<P,T> {}
  // parses left, then if it fails, parses right.
  record Choice<P, T>(Peg<P,T> left, Peg<P,T> right) implements Peg<P,T> {}

  // parse multiple of the same, folded together. List<A> is a bit more expressive 
  // than needed but close enough in java. I think in haskell, the A would be
  // Foldable or something.
  record OneOrMore<P, A, T>(Peg<P, A> peg, Function<List<A>, T> fold) implements Peg<P,T> {}
  

  // the other way to do OneOrMore is
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
  @SuppressWarnings("unchecked")
  default <P, A, R> Peg<P, R> map(Function<T, R> f) {
    return switch (this) {
        case Terminal(var value, var fmap) -> new Terminal<P, R>(
            ((P)value), ((Function<P, T>)fmap).andThen(f));
        case Success(var value) -> new Success<>(f.apply(value));
        case Fail(var message) -> new Fail<>(message);
        case Sequence(var first, var next) ->
          // XXX java type system not powerful enough for this.
          new Sequence<P, A, R>(
              ((Peg<P, A>)first), ((Peg<P, Function<A, T>>)next).map(f::compose));
        case Choice(var left, var right) -> new Choice<>(left.map(f), right.map(f));
        case OneOrMore(var peg, var fold) -> 
          new OneOrMore<>(((Peg<P, A>)peg), ((Function<List<A>, T>)fold).andThen(f));
        // the peg AtomicReference is shared (and eventually set)
        case Ref(var peg, var map) -> new Ref(peg, map.andThen(f));
    }; 
  }

  // helpers

  // convenience for when you just need a marker for whatever got parsed.
  default <P, R> Peg<P, R> as(R r) {
    return map(__ -> r);
  }

  static <P, T> Ref<P, T, T> ref() {
    return new Ref<>(new AtomicReference<>(), Function.identity());
  }

  // fold multiple choices into a linked list, essentially.
  @SuppressWarnings("unchecked")
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

  // extremely efficient string literal parser, character by character.
  static Peg<Character, String> literal(String value) {
    if (value.length() == 0) throw new AssertionError("blank literal");
    Peg<Character,String> first = new Terminal<>(value.charAt(0), String::valueOf);

    // fold characters into a sequence of literals that foldss results back into a string.
    return value.chars()
      .mapToObj(c -> (Peg<Character,String>) new Terminal<>((char)c, String::valueOf))
      .reduce(
          first, 
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

/*
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

class CombiLox {
  // TODO ah, here's where you need the sort of lazy reference. the grammar
  // is circular, though not directly recursive.
  // reference would be a Peg<P,T> subclass that composes any functions mapped over.
  // once reference is set, apply the functions.
  // in interpreter for actual parsing, uh, does it need to know?
  record LazyPeg<U, T>(AtomicReference<Peg<U>> underlying, Function<U, T> mapping) extends Peg<P,T> {
    @Override
    public <R> Peg<R> map(Function<T, R> f) {
        return new LazyPeg<U, R>(underlying, mapping.andThen(f));
    }

    // eventually set the underlying, and uses need to apply the mapping.

  }
  
  static Peg<Expression> ExpressionPeg = new LazyPeg(new AtomicReference<Peg<Expression>>(), (t -> t));

  static Peg<Literal> LiteralPeg = Pegs.choices(
      Pegs.literalString("nil").map(__ -> new LoxNil()),
      Pegs.literalString("true").map(__ -> new LoxTrue()),
      Pegs.literalString("false").map(__ -> new LoxFalse()),

      Pegs.oneOrMore(
      Pegs.choices("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
      .map(digits -> new LoxNumber(parseFloat(digits.join("")))),
      Pegs.surround(Pegs.alpha, "\"").map(LoxString::new)
    );

  static Peg<Unary> UnaryPeg = new Sequence<Unary>(
      Pegs.choices(
        Pegs.literalString("-").map(__ -> UnaryOp.NEGATIVE),
        Pegs.literalString("!").map(__ -> UnaryOp.NOT)),
      ExpressionPeg.map(exp -> op -> new Unary(op, exp)));


  static Peg<Operator> OperatorPeg = Pegs.choices(
      Pegs.literalString("==").map(__ -> EQUALS),
      Pegs.literalString("!=").map(__ -> NEQUALS),
      Pegs.literalString("<").map(__ -> LESS),
      Pegs.literalString("<=").map(__ -> LESSEQUAL),
      Pegs.literalString(">").map(__ -> GREATER),
      Pegs.literalString(">=").map(__ -> GREATEREQUAL),
      Pegs.literalString("+").map(__ -> PLUS),
      Pegs.literalString("-").map(__ -> MINUS),
      Pegs.literalString("*").map(__ -> MULT),
      Pegs.literalString("/").map(__ -> DIV)
    );

  static Peg<Binary> BinaryPeg = new Sequence<Binary>(
      ExpressionPeg,
      new Sequence<Function<Expression, Unary>>(
        OperatorPeg,
        ExpressionPeg.map(rightExp -> op -> leftExp -> new Unary(leftExp, op, rightExp))
        )
      );

  static Peg<Grouping> GroupingPeg = 
    // TODO or sequence, then list.get(1)
    Pegs.surround(ExpressionPeg, "()").map(Grouping::new);

  {
    ExpressionPeg.underlying.set(Pegs.choices(
          LiteralPeg,
          UnaryPeg,
          BinaryPeg,
          GroupingPeg));
  }
}
*/
