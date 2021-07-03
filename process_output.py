import os
import json
import numpy as np
import pandas as pd
import copy

table_input = '''
<tr>
    <td>{product}</td>
    <td>{pve}</td>
    <td>{nve}</td>
  </tr>
'''
base_format = '''
    <html>
<head>
<style>
{css}
</style>
</head>
<body>

<h2>Model Output</h2>
<h3> Top 5 Recommendation out of 20 Recommendation  as per reviews sentiment</h3>
<h3>Username is {Username}</h3>
<table>
  <tr>
    <th>Product</th>
    <th>+ve review %</th>
    <th>-ve review %</th>
  </tr>
  
  {table}
</table>

</body>
</html>

    '''

css = '''
table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}

tr:nth-child(even) {
  background-color: #dddddd;
}
'''


class FormartOutput(object):
    def __init__(self, model):
        self.model = model

    def classifier_output(self, sentence):
        '''
        As this was a highly imbalance dataset,hence we will use fsd to classify the class
        here i m taking 0.60 fsd as min for class 1.i.e +ve
        '''
        doc = self.model.vector_embedding(sentence)
        vector = np.asarray(doc.vector).reshape(1, -1)
        predicted_output = self.model.clf.predict(vector)[0]
        scores = self.model.clf.predict_proba(vector)[0]
        fsd = max(scores) - min(scores)
        if predicted_output == 1 and fsd < 0.6:
            predicted_output = 0
        return predicted_output

    def recommendation_system(self, uid):
        d = self.model.user_final_rating.loc[uid].sort_values(ascending=False)[0:20]
        d = pd.merge(d, self.model.df_reco, left_on='product_id', right_on='product_id', how='left')
        user_product_recommended = []
        user_product = list(set(list(d.product_id)))
        for pid in user_product:
            user_product_recommended.append(
                list(self.model.product_id_vocab.keys())[list(self.model.product_id_vocab.values()).index(pid)])
        return user_product, user_product_recommended

    def process(self, uid):
        user_name = list(self.model.user_id_vocab.keys())[list(self.model.user_id_vocab.values()).index(uid)]
        user_product, user_product_recommended = self.recommendation_system(uid)

        table_output = []
        output = []
        for pid, product in zip(user_product, user_product_recommended):
            reviews = list(self.model.df_reco.reviews_text[self.model.df_reco.product_id == pid])
            sentiment_review = []
            for text in reviews:
                sentiment_review.append(self.classifier_output(text))
            # print("product", product)
            pve = sum(sentiment_review) / len(sentiment_review)
            nve = 1 - pve
            output.append({"product": product, "pve": pve, "nve": nve})
            # table_output.append(table_input.format(product=product, pve=pve, nve=nve))
        output = sorted(output, key=lambda k: k['pve'], reverse=True)
        for data in output[:5]:
            table_output.append(table_input.format(product=data['product'], pve=data['pve'], nve=data['nve']))

        return base_format.format(Username=user_name, table=" ".join(table_output), css=css)


# if __name__ == '__main__':
#     from load_model import LoadModel
#
#     obj = FormartOutput(LoadModel())
#     print(obj.process(4))
