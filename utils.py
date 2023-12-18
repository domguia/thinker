
class MatchCount:
    def __init__(self, size):
        self.match_count = torch.zeros(size, dtype=int)
        self.max = 0
    def update(self, logits, targets):
        acc = (targets == torch.argmax(logits, dim=2)).sum(dim=1).cpu()
        uniques, counts = torch.unique(acc, return_counts=True)
        self.match_count[uniques] += counts
        self.max = uniques.max().item()
    def __str__(self):
        out = ''
        for c in self.match_count[:self.max]:
            out += f"{c:5d} "
        out += " max = %d\n" % (self.max,)
        mean = self.match_count / self.match_count.sum()
        for c in mean[:self.max]:
            out += f" {c:.2f} "
        return out

### test
# match_count = MatchCount(20)
# match_count.update(logits, targets)
# print(match_count)

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    def __call__(self, *args, **kwargs):
        self.__dict__.update(**kwargs)
        args = [item.strip() for items in args for item in items.split(',')]
        self.__dict__.update(**{name: globals()[name] for name in args})
    def __str__(self):
        return self.__dict__.__str__()