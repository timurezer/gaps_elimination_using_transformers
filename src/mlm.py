from naive_recover import *


class TabTransformerMLM(nn.Module):
    def __init__(
            self,
            *,
            categories,
            dim,
            depth,
            heads,
            dim_head=16,
            attn_dropout=0.,
            ff_dropout=0.,
            seed=42,
            mask_mode='single',  # 'single' - missing will be tokenized by the same token, 'different'
            inference=False
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        self.inference = inference
        self.mask_mode = mask_mode
        self.categories = categories
        torch.manual_seed(seed)

        # categories related calculations

        self.num_categories = len(categories)  # len of input sequences
        self.num_unique_categories = sum(categories)  # number of all categories

        # create category embeddings table
        if self.mask_mode == 'single':
            # one special zero token for missing values in all columns
            num_special_tokens = 1

        else:
            # mask_mode == 'different':
            # for each column special missing token
            num_special_tokens = len(categories)
        self.num_special_tokens = num_special_tokens
        # so tokens are encoded as [1 or len(categ)] + [num_unique_categories]

        total_tokens = self.num_unique_categories + num_special_tokens
        self.total_tokens = total_tokens
        # print(categories)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        # (2,3,2,3), num_sp_t = x  --pad--> (x,2,3,2,3) --cumsum[:-1]--> (x, 2+x, 5+x, 7+x)
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]  # cumulative sum
        self.register_buffer('categories_offset', categories_offset)  # save it and use in forward

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        self.classificator = nn.Linear(dim, self.num_unique_categories)   # without special tokens

    def forward(self, x_categ: torch.tensor):
        assert x_categ.shape[-1] == self.num_categories, \
            f'you must pass in {self.num_categories} values for your categories input'

        initial_labels = torch.zeros((x_categ.shape[0], self.num_unique_categories))

        if self.inference:
            cols = torch.where(torch.isnan(x_categ))  # torch.arange(x_categ.shape[0])
            rows = cols[0]
            cols = cols[1]

            # print(cols)
        else:
            rows = torch.arange(x_categ.shape[0])
            cols = torch.tensor(np.random.choice(x_categ.shape[1], x_categ.shape[0]))  # TODO inference
            idx = x_categ[(rows, cols)] + self.categories_offset[cols] - self.num_special_tokens
            initial_labels[(rows, idx)] = 1

        x_categ += self.categories_offset
        x_categ = x_categ.long()
        if self.mask_mode == 'single':
            x_categ[(rows, cols)] = 0
        else:
            x_categ[(rows, cols)] = cols
        x_categ = x_categ.long()

        x = self.transformer(x_categ)

        pred_logits = self.classificator(x[(rows, cols)])
        return pred_logits, initial_labels

