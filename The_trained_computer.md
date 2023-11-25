""" # python comment out ;-)

# The trained computer

We want to train a model that do computation such as 1 + 2 = 3,
what do we need for computation:
1. `input`
2. `computer`
3. `memory`
4. `algorithm`, that give the desired output

in our case `computer` & `algorithm` will be merge in the `model`
the model will be a cross-attention latent based transformers ie. Perciever

`memory` will be intermediates latents

the `algorithm` we decided to learn are (from easy to difficult) :
1. addition
2. multiplication
3. number factorisation
    for variable complexity and highly rely on memmory to reduce computation (close to think process aka LM)

Below the code draft, should be implemented properly
```python
# """

import numpy as np
# MAX_NUMBER = 1e10


vocabulary = \
    #0-255: base number
    [i for i in range(255)] + 
    #256-258: maker
    ['@'] + # for base out : 2, 4, 10, 16, 32, 64, 128, 256
    ['.'] + # comma separator
    ['!'] + # negative number
    ['blank'] + # blank token, for withe space
    # number encoding 16_123 = 312 in base 16
    #        or 16_123.01223 = 312.01223 in base 16
    #258-: operation and trigger
    ['+','-','*','/','%','=',
        '^', # for power
        'output:base',
        'task:factorise',
    ]

def to_base(n, base):
    y = 0
    #... computation
    return y

# def change_base_generator(batch=2, max_number=1000):
#     return np.random.randn(batch, max_number)

def add_mult_generator(batch=2, step=2, max_number=1000, operations='+-*'):
    _step = sum(step)/2 if step is set else step
    numbers = np.random.randn(batch*_step, max_number)
    applied_op = np.random.choice(operations, batch*(_step-1))

    numbers = numbers.reshape(batch, _step)

    def parse_output(numbers, operations):
        y = numbers[0]
        for n,op in zip(numbers,operations):
            if op == '+': y += n
            if op == '-': y -= n
            if op == '*': y *= n # should be priotized
            if op == '/': y /= n # should be priotized
        return y

    def parse_input(numbers):
        inp = numbers[0]
        for n,op in zip(numbers,operations):
            inp += [n, op]
        return inp
    
    x = zip(numbers,applied_op).apply(parse_input, dim=1)
    y = zip(numbers,applied_op).apply(parse_output, dim=1)
    return x, y

PRIME_NUMBERS = []
def factorize_generator(batch=2, step=2, max_number=1000):
    """
        step : number or (number, number)
                number : number of product step

    """
    prime_numbers = [i for i in PRIME_NUMBERS if i<=max_number]
    if isinstance(step, set):
        # _step = np.random.choice(range(*step))
        _step = sum(step)/2
    else
        _step = step
    
    factors = np.random.choice(prime_numbers, batch*_step, repeat=True)
    factors = factors.reshape(batch,step)
    factors = factors.sort(dim=1)

    def parse_input(numbers):
        prod = 1
        for n in numbers:
            prod *= n
        return prod

    def parse_output(numbers):
        n = len(numbers)
        out = numbers[0:1]
        power = 0
        for i in range(1,n):
            if numbers[i-1] == numbers[i]:
                if power == 0:
                    out += ['^']
                power += 1
            else:
                if power > 0:
                    out.append(power)
                out.append(numbers[i])
                out += ['*']
        return out
        
    x = factors.apply(parse_input, dim=1)
    y = factors.apply(parse_output, dim=1)
    return x, y
    
def apply_to_base(number, base=10):
    return [ to_base(x,base) if isinstance(x, number) else x for x in number ]


task_scheme = {
    'input':{
        'base':{
            'value' : [2, 4, 10, 16], # 32, 64, 128, 256],
            'distribution' : [1,10], # sampling distribution probability
            # probability of the first element (2) is 1/10 the probability of
            # having the last element and probability should progress exponentialy
        },
        'max_lenght' : 1024,
        'overflow_strategy' : 'increase_base',
    },
    'output': '[input]',
    'generator': {
        'factorize_generator':{
            'step': range(2,10),
        },
        'add_mult_generator':{
            'step': range(2,10),
            'operations': '+-*', # +-*
        },
        'arguments':{
            'batch': 32,
            'max_number': 1_000_000_000,
        }
    }
}

def get_data(task_scheme):
    if task_scheme['output'] == '[input]':
        task_scheme['output'] = task_scheme[input]
    
    operations = task_scheme['generator']

    factorize_generator(batch, step, max_number)
    

# # number of call step for the model should be evaluated considering task scheme and memory usage
# params = dict(
#     # data param
#     batch_size = (1, 4, 8, 32),
#     input_lenght = (16, 64, 128, 256, 512, 1024),
#     output_lenght = (16, 64, 128, 256, 512, 1024),

#     # model run param
#     steps = (1, 4, 8, 16, 32, 64, 128),
#     latent = (4, 8, 16, 32, 64, 128),
#     memory_context = (16, 32, 64, 128),

#     # model weight param
#     dim = (32, 64, 128, 256, 512, 1024)
#     n_layers = (1,2,3)
#     n_heads = 8
#     # head_dim = 8
#     # hidden_dim = ()
# )

config = dict(
    mem_cache_size = 512,
    input_cache_size = 512,
    latent_size = 512,
)

query_token = {
    'probe':0,
    'start_decode':1,
}

config = dict(
    dim = 32,
    n_layers = 2,
    n_heads = 8,

    vocab = 256 + 16,
    latent = 256,
)
model = Th1nker(config)

dataloader = init_dataloader(task_scheme)
for x,y in dataloader:
    model.init_kv_cache()
    model.init_latent()
    for i in range(step):
        model.read_input(x, latent_to_memory=True)
        # model.latent_to_memory()
        # model.memory_lookup()
        model.process()
        model.write_out(y)

for 


```
