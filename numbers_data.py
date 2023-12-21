import numpy as np
import torch
from thinker_model import CfgNode
# MAX_NUMBER = 1e10

MAX_BASE = 16 + 1 #256
VOCABULARY = sum([
    #0-255: base number
    [i for i in range(MAX_BASE)],
    #256-258: maker
    ['@'], # for base out : 2, 4, 10, 16, 32, 64, 128, 256
    ['.'], # comma separator
    ['!'], # negative number
    ['blank'], # blank token, for withe space
    # number encoding 16_123 = 312 in base 16
    #        or 16_123.01223 = 312.01223 in base 16
    #258-: operation and trigger
    ['+','-','*','/','%','='],
    ['^'], # for power
    ['@='], # output base
    # :task
    [':fact'], # trigger for factorising task
    [':addmul'], # addition, multiplication task
],[])
VOCABULARY_DICT = {token:i for i, token in enumerate(VOCABULARY)}
VOCABULARY_DICT.update({str(i):i for i in range(MAX_BASE)}) # add str version

def to_base(n, base):
    digits = []
    if n<0:
        digits.append('!')
        n = -n
    if base < 2:
        raise ValueError("Input values base must be at least 2.")
    if n == 0: return [0]
    
    while n > 0:
        digits.append(n % base)
        n //= base
    # return digits[::-1]
    return digits # reversed digits are ok

# def change_base_generator(batch=2, max_number=1000):
#     return np.random.randn(batch, max_number)

def add_mult_generator(batch=2, step=2, max_number=1000, operations='+-*', operation_dist=None, task_id=None, **kwargs):
    _step = sum(step)/2 if step is set else step
    numbers = np.random.randint(max_number, size=(batch, _step))
    applied_op = np.random.choice(list(operations), size=(batch,  _step), p=operation_dist)

    def parse_output(numbs,ops):
        y = numbs[0]
        for op, n in zip(ops[1:], numbs[1:]):
            if op == '+': y += n
            if op == '-': y -= n
            if op == '*': y *= n # should be priotized
            if op == '/': y //= n # only int divison # should be priotized
        return [y]

    def parse_input(numbs,ops):
        inp = [numbs[0]]
        for op, n in zip(ops[1:], numbs[1:]):
            inp += [op, n]
        return inp

    x = [parse_input(numbers[i], applied_op[i]) for i in range(batch)]
    y = [parse_output(numbers[i], applied_op[i]) for i in range(batch)]
    return x, y

def generate_primes(n):
    def prime(i, primes):
        for prime in primes:
            if not (i == prime or i % prime):
                return False
        primes.append(i)
        return i

    primes = [2]
    i, p = 3, 1
    while True:
        if prime(i, primes):
            p += 1
            if p == n:
                return primes
        i += 2

MAX_PRIME = 1000
PRIME_NUMBERS = []
PRIME_NUMBERS = generate_primes(MAX_PRIME)

def factorize_generator(batch=2, step=2, max_number=None, n_first_prime=100, **kwargs):
    """
        step : number or (number, number)
                number : number of product step

    """
    if max_number: 
        for i,n in enumerate(PRIME_NUMBERS):
            if n>max_number:
                break
        n_first_prime = i # little bug, skip the last idx

    if isinstance(step, set):
        # _step = np.random.choice(range(*step))
        _step = sum(step)/2
    else:
        _step = step
    
    factors = np.random.choice(PRIME_NUMBERS[:n_first_prime], size=(batch,_step), replace=True)
    factors.sort(axis=1)
    
    def parse_input(numbers):
        prod = 1
        for n in numbers:
            prod *= n
        return [prod]

    def parse_output(numbers):
        n = len(numbers)
        out = numbers[0:1].tolist()
        power = 0
        for i in range(1,n):
            if numbers[i-1] == numbers[i]:
                if power == 0:
                    out += ['^']
                    power = 1
                power += 1
            else:
                if power > 0:
                    out.append(power)
                    power = 0
                out += ['*']
                out.append(numbers[i])
        if power > 0:
            out.append(power)
        return out
        
    x = [parse_input(facs)  for facs in factors]
    y = [parse_output(facs) for facs in factors]
    return x, y
    
def apply_to_base(n, base=10):
    if isinstance(n, int): return to_base(n, base)
    return sum([ [x] if isinstance(x, str) else to_base(x,base) for x in n ],[])

def token_encoder(inputs, voc_dict=VOCABULARY_DICT):
    if isinstance(inputs, str): return voc_dict[inputs]
    return [voc_dict[x] for x in inputs]

def token_decoder(inputs, voc_list=VOCABULARY):
    return [voc_list[x] for x in inputs]

TASK_SCHEME = {
    'input':{ # input base 
        'base':{
            'values' : list(range(2,16+1)), #[2, 4, 10, 16], # 32, 64, 128, 256],
            'distribution' : None, # [0.1, 0.2, 0.3, 0.4], # sampling distribution probability
            # probability of the first element (2) is 1/10 the probability of
            # having the last element and probability should progress exponentialy
        },
        'max_length' : 41, # input max lenght after tokenisation
        'overflow_strategy' : 'increase_base',
    },
    'output': { # input base 
        'base':{
            'values' : list(range(2,16+1)), #[2, 4, 10, 16],
            'distribution' : None, # [0.1, 0.2, 0.3, 0.4],
        },
        'max_length' : 22
    },
    'tasks': {
        'add_mult_generator':{
            'task_id': ':fact',
            'step': [2,3],
            'operations': '+-*/',
            'operation_dist':[1,.0,.0,.0],
            # 'operation_dist':[.5,.5,.0,.0],
            # 'operation_dist':[.1,.1,.4,.4],
        },
        'factorize_generator':{
            'task_id': ':fact',
            'step': [2,10], # should sample value in the given range
            'n_first_prime': 100,
        },
        'shared_args':{
            'batch': 2,
            'max_number': 1_000_000,
            'task_probabilities': [1.0,0.0],
        }
    }
}


import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

def sample_base(values, batch=None, distribution=None):
    # Normalize the distribution to sum to 1
    # distribution = np.array(distribution)
    # distribution = distribution / distribution.sum()
    # Sample a base according to the probability distribution
    base = np.random.choice(values, size=batch, p=distribution)
    return base

# Update the task_scheme definition with an actual distribution
# task_scheme['input']['base']['distribution'] = [1 / (2 ** i) for i, _ in enumerate(task_scheme['input']['base']['values'])]

class NumbersCopyDataset(IterableDataset):
    count = 0
    accuracy = 0
    challenge_factor = 1
    
    def __init__(self, vocab_size, seq_len, batch, uniform_len=False, task=None):
        self.vocab_size = vocab_size
        self.batch = batch
        self.seq_len = seq_len
        # task
        self.uniform_len = uniform_len
        self.task = task

    @classmethod
    def update_accuracy(cls, accuracy):
        cls.accuracy = accuracy
        
    @classmethod
    def update_challenge_factor(cls, challenge_factor):
        cls.challenge_factor = challenge_factor
    
    @classmethod
    def get_challenge_factor(cls):
        return cls.challenge_factor

    @classmethod
    def reset(cls, count=0):
        cls.count = count

    @classmethod
    def incr(cls, count=1):
        cls.count += count

    @classmethod
    def get_dist(cls, count=0):
        cls.count = count

    def __iter__(self):
        seq_len, batch = self.seq_len, self.batch
        while True:
            NumbersCopyDataset.incr(batch)
            accuracy = NumbersCopyDataset.accuracy
            target_len = accuracy*seq_len

            NumbersCopyDataset.update_challenge_factor(target_len/seq_len)
            
            if 'progressive_copy' == self.task:
                mask_target_len = seq_len - target_len
                # :-) basic implementation of prgoressive/currilum learning
                
                # build mask
                # _, mu, _ -> 0, target_len, seq_len
                s = torch.normal(0, 1, (batch,))
                s = 1 + s / (max(s)-min(s))
                s = s * mask_target_len
                s = torch.clip(s, 0, seq_len)
                mask_len = s.int() - 1
                # mask_len = torch.randint(0, seq_len-3, (batch,)) # uniform generation
                mask = torch.arange(seq_len)[None,:].expand(batch, -1) > mask_len[:,None]

                x = torch.randint(0, self.vocab_size, (batch, seq_len))

                x = torch.where(mask, x, 0)
                y = x # just copy
            # run tasks
            if None == self.task:
                y = x #.clone()
            if 'sort' == self.task:
                y = x.sort(dim=1)[0]
            if 'flip' == self.task:
                y = x.flip(dims=(1,))
            if 'roll' == self.task:
                y = torch.Tensor(np.apply_along_axis(lambda a: np.roll(a, a[0]), 1, x))
                # y = x.roll(x[0].item(), 0) # for non batched
            if 'roll_num' == self.task:
                y = (x + x[:,0]) % self.high
            if 'jump' == self.task:
                idx = x[:,0]
                for i in range(16): # max 16 jumps
                    idx += x[:, idx]
                jump = x[:,0] # number of jump
                y = idx[:, jump]
            if 'cumsum' == self.task:
                # y = x.clone()
                # for i in range(x.size(1)):
                #     y[:,i:] = (y[:,i:] + y[i:,i]) % self.high # faster that position base flip
                y = torch.cumsum(x, dim=1) % self.vocab_size
            yield x, y

'''
# thinking

how to compute loss?
- at each step
- if possible has intermediate target
Note: end to end is great but soft end to end is faster to learn


How do we handle the noisy property of text?
- for text final output use autoregressive decoding, because the model should be inform about the previous token to correctly decode the coming one, but de autogressive should be keep to the smallest amount of computation eg. with 1 layers
- intermediate decoding could be done to evaluate if the model intermediate state get closer to the solution
- give inductive bias that will direct the model : summary, keyword, ... (inductive input tokens get optimization on expected output)

'''

class NumbersComputeDataset(IterableDataset):
    def __init__(self, cfg):
        # self.task_scheme = task_scheme
        self.cfg = cfg
    
    @staticmethod
    def get_vocabulary():
        return VOCABULARY_DICT
    
    @staticmethod
    def get_vocabulary_size():
        return len(VOCABULARY)

    def __iter__(self):

        while True:
            # TODO: should samples task base on task_scheme definition
            # samples = factorize_generator(batch=2, step=6, n_first_prime=10)
            # samples = add_mult_generator(batch=2, step=4, max_number=1000, operations='+-*/', operation_dist=[0.1,0.1,0.4,0.4])
            samples = add_mult_generator(**self.cfg.__dict__)
            task_id = ':addmul'

            # TODO: should samples in_base and out_base looking at task_scheme definition
            # Sample input and output bases
            in_bases = sample_base(self.cfg.in_bases, self.cfg.batch, self.cfg.in_bases_dist)
            out_bases = sample_base(self.cfg.out_bases, self.cfg.batch, self.cfg.out_bases_dist)

            all_x = []
            all_y = []
            for x,y, in_base,out_base in zip(*samples, in_bases, out_bases):
                x = apply_to_base(x, in_base)
                y = apply_to_base(y, out_base)
                
                # add input trriger
                x = [in_base, '@'] + x + [task_id, '@=', out_base]

                # tokenize
                x = token_encoder(x)
                y = token_encoder(y)

                # save
                all_x.append(x)
                all_y.append(y)

            # get max lenght for padding
            x_max_len = max([len(x) for x in all_x])
            y_max_len = max([len(y) for y in all_y])

            # apply padding
            blank_token = token_encoder('blank')
            all_x = [     [blank_token]*(x_max_len-len(x)) + x for x in all_x] # pad begin
            all_y = [ y + [blank_token]*(y_max_len-len(y))     for y in all_y] # pad end

            all_x = np.stack(all_x)#, dtype=np.int16)
            all_y = np.stack(all_y)#, dtype=np.int16)
                
            yield torch.IntTensor(all_x), torch.IntTensor(all_y)

    

if __name__ == '__main__':

    # cfg = CfgNode(batch=5, step=4, max_number=1000, operations='+-*/', operation_dist=[0.1,0.1,0.4,0.4],
    #            in_bases=[2,4,8,16], in_bases_dist=None,
    #            out_bases=[2,4,8,16], out_bases_dist=[.1,.2,.3,.4])
    # dataset = NumbersComputeDataset(cfg)
    
    dataset = NumbersCopyDataset(vocab_size=16+1, seq_len=7, batch=3, task='progressive_copy')
    NumbersCopyDataset.update_accuracy(.0)
    x, y = next(iter(dataset))
    print(f"Batch : x={x.shape}, y={y.shape}")
    print(x)
    print('-'*9)
    print(y)
    
    # dataset = NumbersComputeDataset(TASK_SCHEME)

    # batch_size = task_scheme['tasks']['shared_args']['batch']  # Set the batch size based on the scheme
    # batch_size = 32
    # dataloader = DataLoader(dataset, batch_size=batch_size)

    # # Example of iterating through the data loader (add stopping criteria as needed)
    # for batch_idx, (x, y) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}: x={x.shape}, y={y.shape}")
    #     if batch_idx >= 1:  # Stop after 2 batches for this example
    #         break