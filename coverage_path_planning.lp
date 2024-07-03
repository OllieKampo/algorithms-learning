% Graph nodes
% #program declare_domain(total_nodes, start_node).

#const total_nodes = 20.
#const start_node = 1.

% Graph nodes
node(1..total_nodes).

{ cycle(X, Y) } :- node(X), node(Y), X != Y.

% Each node is visited as few times as possible.
not_once(Y) :- not #count { X : cycle(X, Y), node(X) } = 1, node(Y).
not_once(X) :- not #count { Y : cycle(X, Y), node(Y) } = 1, node(X).
:~ not_once(X), node(X). [1 @ 2, X]

% All nodes must be visited at least once.
in_cycle(X) :- cycle(start_node, X), node(X).
in_cycle(Y) :- in_cycle(X), cycle(X, Y), node(X), node(Y).
:~ not in_cycle(X), node(X). [1 @ 1, X]

% Must return to start position.
returns_to_start :- cycle(X, start_node), node(X).
:- not returns_to_start.
