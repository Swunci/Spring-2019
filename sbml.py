# Steven Chen 110884657


class Node:
    def __init__(self):
        print("init node")

    def evaluate(self):
        return 0

    def execute(self):
        return 0


class NumberNode(Node):
    def __init__(self, v):
        if '.' in v:
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self):
        return self.value


class BopNode(Node):
    def __init__(self, op, node1, node2):
        self.Node1 = node1
        self.Node2 = node2
        self.op = op

    def evaluate(self):
        try:
            if self.op == '+':
                return self.Node1.evaluate() + self.Node2.evaluate()
            elif self.op == '-':
                return self.Node1.evaluate() - self.Node2.evaluate()
            elif self.op == '*':
                return self.Node1.evaluate() * self.Node2.evaluate()
            elif self.op == '/':
                return self.Node1.evaluate() / self.Node2.evaluate()
            elif self.op == '**':
                return self.Node1.evaluate() ** self.Node2.evaluate()
            elif self.op == 'div':
                return self.Node1.evaluate() // self.Node2.evaluate()
            elif self.op == 'mod':
                return self.Node1.evaluate() % self.Node2.evaluate()
            elif self.op == '<':
                return self.Node1.evaluate() < self.Node2.evaluate()
            elif self.op == '<=':
                return self.Node1.evaluate() <= self.Node2.evaluate()
            elif self.op == '==':
                return self.Node1.evaluate() == self.Node2.evaluate()
            elif self.op == '<>':
                return self.Node1.evaluate() != self.Node2.evaluate()
            elif self.op == '>':
                return self.Node1.evaluate() > self.Node2.evaluate()
            elif self.op == '>=':
                return self.Node1.evaluate() >= self.Node2.evaluate()
            elif self.op == 'andalso':
                return self.Node1.evaluate() and self.Node2.evaluate()
            elif self.op == 'orelse':
                return self.Node1.evaluate() or self.Node2.evaluate()
        except Exception:
            raise MySemanticError(Exception)


# For my String Node, when it is evaluated, it will be enclosed with single quotes
class StringNode(Node):
    def __init__(self, v):
        # The single/double quotes at the end of the string are removed
        self.string = v[1:-1]

    def evaluate(self):
        return self.string


class EmptyListNode(Node):
    def __init__(self):
        self.list = []

    def evaluate(self):
        return self.list


class ListNode(Node):
    def __init__(self, v):
        self.list = [v]

    def evaluate(self):
        return_list = []
        for x in self.list:
            return_list = return_list + [x.evaluate()]
        return return_list


# My TupleNode keeps a list of the values, when it is evaluated it will convert the list to a tuple
class TupleNode(Node):
    def __init__(self, v):
        self.list = [v]

    def evaluate(self):
        return_list = []
        for x in self.list:
            return_list = return_list + [x.evaluate()]
        return tuple(return_list)


class HashTagNode(Node):
    def __init__(self, index, tuple):
        self.indexNode = index
        self.tupleNode = tuple

    def evaluate(self):
        try:
            return self.tupleNode.evaluate()[self.indexNode.evaluate() - 1]
        except Exception:
            raise MySemanticError(Exception)


class IndexingNode(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def evaluate(self):
        a = self.a.evaluate()
        b = self.b.evaluate()
        try:
            return a[b]
        except Exception:
            raise MySemanticError(Exception)


class InNode(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self):
        a = self.a.evaluate()
        b = self.b.evaluate()
        try:
            return a in b
        except Exception:
            raise MySemanticError(Exception)


class ConcatNode(Node):
    def __init__(self, h, t):
        self.h = h
        self.t = t

    def evaluate(self):

        h = self.h.evaluate()
        t = self.t.evaluate()
        try:
            return [h] + t
        except Exception:
            raise MySemanticError(Exception)


class BooleanNode(Node):
    def __init__(self, boolean):
        if boolean == 'True':
            self.boolean = True
        else:
            self.boolean = False

    def evaluate(self):
        return self.boolean


class NotNode(Node):
    def __init__(self, node):
        self.node = node

    def evaluate(self):
        try:
            return not self.node.evaluate()
        except Exception:
            raise MySemanticError(Exception)


class PrintNode(Node):
    def __init__(self, v):
        self.Node = v

    def evaluate(self):
        print(self.Node.evaluate())


class EmptyBlockNode(Node):
    def __init__(self):
        self.nothing = 0

    def execute(self):
        self.nothing = 1


class BlockNode(Node):
    def __init__(self, s1):
        self.statementList = s1

    def execute(self):
        for statement in self.statementList:
            if isinstance(statement, BlockNode) or isinstance(statement, EmptyBlockNode):
                statement.execute()
            else:
                statement.evaluate()


class VariableNode(Node):
    def __init__(self, name):
        self.variableName = name

    def evaluate(self):
        try:
            return variableDictionary[self.variableName]
        except Exception:
            raise MySemanticError(Exception)


class AssignmentNode(Node):
    def __init__(self, variable_node, value_node):
        self.variableNode = variable_node
        self.valueNode = value_node

    def evaluate(self):
        try:
            if isinstance(self.variableNode, IndexingNode):
                if isinstance(self.variableNode.a, IndexingNode):
                    if isinstance(self.variableNode.a.a, VariableNode):
                        variableDictionary[self.variableNode.a.a.variableName][self.variableNode.a.b.evaluate()][
                            self.variableNode.b.evaluate()] = self.valueNode.evaluate()
                    else:
                        raise MySemanticError(Exception)
                elif isinstance(self.variableNode.a, VariableNode):
                    variableDictionary[self.variableNode.a.variableName][self.variableNode.b.evaluate()] = self.valueNode.evaluate()
                else:
                    raise MySemanticError(Exception)
            else:
                variableDictionary[self.variableNode.variableName] = self.valueNode.evaluate()
        except Exception:
            raise MySemanticError(Exception)


class IfStatementNode(Node):
    def __init__(self, expression, block):
        self.expression = expression
        self.block = block

    def evaluate(self):
        if self.expression.evaluate():
            self.block.execute()


class IfElseStatementNode(Node):
    def __init__(self, expression, block1, block2):
        self.expression = expression
        self.block1 = block1
        self.block2 = block2

    def evaluate(self):
        if self.expression.evaluate():
            self.block1.execute()
        else:
            self.block2.execute()


class WhileStatementNode(Node):
    def __init__(self, expression, block):
        self.expression = expression
        self.block = block

    def evaluate(self):
        while self.expression.evaluate():
            self.block.execute()


class FunctionNode(Node):
    def __init__(self, id, parameters, block, output):
        self.id = id
        self.parameters = parameters
        self.block = block
        self.output = output

    def execute(self):
        self.block.execute()
        return self.output


class FunctionCallNode(Node):
    def __init__(self, id, args):
        self.id = id
        self.args = args

    def evaluate(self):
        global variableDictionary
        global functionDictionary
        function_node = functionDictionary[self.id]
        new_dictionary = {}
        for i in range(len(function_node.parameters)):
            new_dictionary[function_node.parameters[i].variableName] = self.args[i].evaluate()
        old_dictionary = variableDictionary
        variableDictionary = new_dictionary
        function_node.execute()
        result = function_node.output.evaluate()
        variableDictionary = old_dictionary
        return result


class ProgramNode(Node):
    def __init__(self, functions, block):
        self.functions = functions
        self.block = block

    def execute(self):
        self.block.execute()


variableDictionary = {}
functionDictionary = {}

reserved = {
    'if' : 'IF',
    'else' : 'ELSE',
    'while' : 'WHILE',
    'print' : 'PRINT',
    'True' : 'TRUE',
    'False' : 'FALSE',
    'not' : 'NOT',
    'andalso' : 'AND',
    'orelse' : 'OR',
    'mod' : 'MOD',
    'div' : 'DIV',
    'in' : 'IN',
    'fun' : 'FUN'

}

tokens = [
    'LPAREN', 'RPAREN',
    'NUMBER', 'eNotation',
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'POWER',
    'STRING', 'SEMICOLON', 'COMMA', 'LBRACKET', 'RBRACKET', 'HASHTAG',
    'CONCAT',
    'LESS_THAN', 'LESS_THAN_EQUAL', 'EQUAL', 'NOT_EQUAL', 'GREATER_THAN', 'GREATER_THAN_EQUAL',
    'LCURLY', 'RCURLY',
    'ASSIGN', 'ID'
] + list(reserved.values())

# Tokens

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_SEMICOLON = r';'
t_COMMA = r','
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_HASHTAG = r'\#'
t_POWER = r'\*\*'
t_CONCAT = r'::'
t_LESS_THAN = r'<'
t_LESS_THAN_EQUAL = r'<='
t_EQUAL = r'=='
t_NOT_EQUAL = r'<>'
t_GREATER_THAN = r'>'
t_GREATER_THAN_EQUAL = r'>='
t_LCURLY = r'{'
t_RCURLY = r'}'
t_ASSIGN = r'='


def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')    # Check for reserved words
    if t.type == 'ID':
        t.value = VariableNode(t.value)
    return t


def t_eNotation(t):
    r'-?\d*\.?\d+e[+-]?\d+'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t


def t_NUMBER(t):
    #r'(-?\d*(\d\.|\.\d)\d* | \d+)'
    r'(-?\d * (\d\.| \.\d)\d * | \d +)'
    try:
        t.value = NumberNode(t.value)
    except ValueError:
        print("Integer value too large %d", t.value)
        t.value = 0
    return t


def t_STRING(t):
    # Matches any string surrounded by single/double quotes
    r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    t.value = StringNode(t.value)
    return t


# Ignored characters
t_ignore = " \t\n"


def t_error(t):
    raise MySyntaxError(Exception)


# Build the lexer
import ply.lex as lex

lex.lex(debug=0)

# Parsing rules
precedence = (
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'NOT'),
    ('left', 'LESS_THAN', 'LESS_THAN_EQUAL', 'EQUAL', 'NOT_EQUAL', 'GREATER_THAN', 'GREATER_THAN_EQUAL'),
    ('right', 'CONCAT'),
    ('left', 'IN'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE', 'DIV', 'MOD'),
    ('right', 'POWER'),
    ('left', 'LBRACKET', 'RBRACKET'),
    ('left', 'HASHTAG'),
    ('left', 'LPAREN', 'RPAREN'),
)


def p_program(t):
    '''
    program : functions block
    '''
    t[0] = ProgramNode(t[1], t[2])


def p_functions(t):
    '''
    functions : functions function
    '''
    t[0] = functionDictionary[t[2].id] = t[2]


def p_functions2(t):
    '''
    functions : function
    '''
    t[0] = functionDictionary[t[1].id] = t[1]


def p_function(t):
    '''
    function : FUN ID LPAREN parameters RPAREN ASSIGN block expression SEMICOLON
    '''
    t[0] = FunctionNode(t[2].variableName, t[4], t[7], t[8])


def p_parameters(t):
    '''
    parameters : parameters COMMA ID
    '''
    t[1].append(t[3])
    t[0] = t[1]


def p_parameters_2(t):
    '''
    parameters : ID
    '''
    t[0] = [t[1]]


def p_expression_function_call(t):
    '''
    expression : function_call
    '''
    t[0] = t[1]


def p_function_call(t):
    '''
    function_call : ID LPAREN args RPAREN
    '''
    t[0] = FunctionCallNode(t[1].variableName, t[3])


def p_args(t):
    '''
    args : args COMMA expression
    '''
    t[1].append(t[3])
    t[0] = t[1]


def p_args2(t):
    '''
    args : expression
    '''
    t[0] = [t[1]]


def p_block(t):
    '''
     block : LCURLY statement_list RCURLY
    '''
    t[0] = BlockNode(t[2])


def p_empty_block(t):
    '''
    block : LCURLY RCURLY
    '''
    t[0] = EmptyBlockNode()


def p_statement_list(t):
    '''
     statement_list : statement_list statement
    '''
    t[0] = t[1] + [t[2]]


def p_statement_list_val(t):
    '''
    statement_list : statement
    '''
    t[0] = [t[1]]


def p_statement_block(t):
    '''
    statement : block
    '''
    t[0] = t[1]


def p_if_statement(t):
    '''
    statement : IF LPAREN expression RPAREN block
    '''
    t[0] = IfStatementNode(t[3], t[5])


def p_if_else_statement(t):
    '''
    statement : IF LPAREN expression RPAREN block ELSE block
    '''
    t[0] = IfElseStatementNode(t[3], t[5], t[7])


def p_while_statement(t):
    '''
    statement : WHILE LPAREN expression RPAREN block
    '''
    t[0] = WhileStatementNode(t[3], t[5])


def p_statement_assign(t):
    '''
    statement : ID ASSIGN expression SEMICOLON
              | INDEXING ASSIGN expression SEMICOLON
    '''
    t[0] = AssignmentNode(t[1], t[3])


def p_statement_print(t):
    '''statement : PRINT LPAREN expression RPAREN SEMICOLON'''
    t[0] = PrintNode(t[3])


def p_statement(t):
    '''statement : expression SEMICOLON'''
    t[0] = t[1]


def p_expression_parenthesized_expression(t):
    '''expression : LPAREN expression RPAREN'''
    t[0] = t[2]


def p_expression_indexing(t):
    '''expression : INDEXING'''
    t[0] = t[1]


def p_indexing(t):
    '''INDEXING : expression LBRACKET expression RBRACKET'''
    t[0] = IndexingNode(t[1], t[3])


def p_concat(t):
    '''expression : expression CONCAT expression'''
    t[0] = ConcatNode(t[1], t[3])


def p_in(t):
    '''expression : expression IN expression'''
    t[0] = InNode(t[1], t[3])


def p_empty_list(t):
    '''expression : LBRACKET RBRACKET'''
    t[0] = EmptyListNode()


def p_list(t):
    '''expression : in_list RBRACKET'''
    t[0] = t[1]


def p_in_list(t):
    '''in_list : LBRACKET expression'''
    t[0] = ListNode(t[2])


def p_in_list2(t):
    '''in_list : in_list COMMA expression'''
    t[1].list.append(t[3])
    t[0] = t[1]


def p_expression_tuple(t):
    '''expression : tuple'''
    t[0] = t[1]


def p_hashtag(t):
    '''expression : HASHTAG expression LPAREN tuple RPAREN
                  | HASHTAG expression LPAREN ID RPAREN
                  | HASHTAG expression LPAREN expression RPAREN'''
    t[0] = HashTagNode(t[2], t[4])


def p_tuple(t):
    '''tuple : in_tuple RPAREN'''
    t[0] = t[1]


def p_in_tuple(t):
    '''in_tuple : LPAREN expression COMMA expression'''
    t[0] = TupleNode(t[2])
    t[0].list.append(t[4])


def p_in_tuple2(t):
    '''in_tuple : in_tuple COMMA expression'''
    t[1].list.append(t[3])
    t[0] = t[1]


def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression
                  | expression DIV expression
                  | expression MOD expression
                  | expression LESS_THAN expression
                  | expression LESS_THAN_EQUAL expression
                  | expression EQUAL expression
                  | expression NOT_EQUAL expression
                  | expression GREATER_THAN expression
                  | expression GREATER_THAN_EQUAL expression
                  | expression AND expression
                  | expression OR expression'''
    t[0] = BopNode(t[2], t[1], t[3])


def p_expression_logical_operator(t):
    '''expression : NOT expression'''
    t[0] = NotNode(t[2])


def p_expression_factor(t):
    '''expression : factor'''
    t[0] = t[1]


def p_expression_unary(t):
    '''expression : MINUS expression
                  | PLUS expression'''
    t[0] = NumberNode(str(-1 * t[2].evaluate()))


def p_expression_eNotation(t):
    '''expression : eNotation'''
    t[0] = t[1]


def p_factor_number(t):
    '''factor : NUMBER'''
    t[0] = t[1]


def p_expression_string(t):
    '''expression : STRING'''
    t[0] = t[1]


def p_boolean(t):
    '''expression : TRUE
                  | FALSE '''
    t[0] = BooleanNode(t[1])


def p_ID(t):
    '''
    expression : ID
    '''
    t[0] = t[1]


def p_error(t):
    raise MySyntaxError(Exception)


import ply.yacc as yacc

yacc.yacc(debug=0)


import sys


class MySyntaxError(Exception):
    pass


class MySemanticError(Exception):
    pass


try:
    file_name = sys.argv[1]
    #for x in range(1 ,12):
    #    print(x, end=" ")
    #   file_name = "input" + str(x) + ".txt"
        #file_name = "atest.txt"
    with open(file_name, 'r') as f:
        data = f.read().replace('\n', '')
        try:
            lex.input(data)
            while True:
                token = lex.token()
                if not token: break
                #print(token)
            root = yacc.parse(data)
            root.execute()
            #print(root)
        except MySyntaxError:
            print("SYNTAX ERROR")
        except MySemanticError:
            print("SEMANTIC ERROR")
except FileNotFoundError:
    print("No file found")

