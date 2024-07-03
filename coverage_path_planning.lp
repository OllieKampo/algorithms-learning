% Graph nodes
% #program declare_domain(total_nodes, start_node).

#const total_nodes = 5.
#const start_node = 1.
#const max_steps = 10.

blocked(2, X) :- X = 1..total_nodes, X != 3.
blocked(X, Y) :- blocked(Y, X).

% Graph nodes
node(1..total_nodes).

% Generate all valid arcs of the graph.
arc(X, Y) :- node(X), node(Y), not blocked(X, Y), X != Y.

% Generate the cycle.
{ cycle(X, Y) } :- arc(X, Y), node(X), node(Y), X != Y.
:~ cycle(X, Y), node(X), node(Y). [1 @ 1, X, Y]

% Each node is visited as few times as possible.
not_once(Y) :- not #count { X : cycle(X, Y), node(X) } = 1, node(Y).
not_once(X) :- not #count { Y : cycle(X, Y), node(Y) } = 1, node(X).
:~ not_once(X), node(X). [1 @ 2, X]

% All nodes must be visited at least once.
in_cycle(X) :- cycle(start_node, X), node(X).
in_cycle(Y) :- in_cycle(X), cycle(X, Y), node(X), node(Y).
:~ not in_cycle(X), node(X). [1 @ 3, X]

% Must return to start position.
returns_to_start :- cycle(X, start_node), node(X).
:- not returns_to_start.

% Construct the final path.
path(1, start_node, X) :- cycle(start_node, X), node(X).
{ path(N + 1, X, Y) } :- path(N, Z, X), cycle(X, Y), node(X), node(Y), node(Z), N = 1..max_steps.
:- not #count { N : path(N, X, Y), N = 1..max_steps } = 1, cycle(X, Y).
:~ path(N, X, Y), node(X), node(Y), N = 1..max_steps. [1 @ 0, N, X, Y]
:- path(N, start_node, X), N != 1, node(X).
