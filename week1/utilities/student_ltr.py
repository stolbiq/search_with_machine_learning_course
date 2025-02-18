import xgboost as xgb
from xgboost import plot_importance, plot_tree


##### Step 3.a
'''
Implement an XGBoost training function.  Called from utilities/xgb_utils.py.

This should be verey similar to the how training is done in the LTR toy program.

:param str xgb_train_data: The file path to the training data you should train with
:param int num_rounds: The number of rounds the training process should undertake before terminating.
:param dictionary xgb_params The XGBoost configuration parameters, such as the objective function, e.g. {'objective': 'reg:logistic'} 
'''
def train(xgb_train_data, num_rounds=5, xgb_params=None ):
    # print("IMPLEMENT ME: xgb train")
    dtrain = xgb.DMatrix(f'{xgb_train_data}?format=libsvm')
    print("Training XG Boost")
    bst = xgb.train(xgb_params, dtrain, num_rounds)
    return bst



##### Step 3.b:
'''
We need to log/extract features for our training data set.  This is a critical step in the 
LTR process where we request features (from the featureset we definied) of all of the documents that match a given query from OpenSearch.
For instance, we might get back the TF-IDF score for a document given our query.

Called from ltr_utils.py

Unlike in the toy LTR example where we issued a single query to OpenSearch per query-document pair, you should be able to retrieve 
all features for all documents in a single query.  See the course content for more details. 

:param str query: The user query to run
:param array doc_ids: An array/list of doc ids that users have clicked on for this query. 
:param str click_prior_query: A query string representing what documents were clicked on in the past.  Not used until the 'Using Prior Query History' section of the project.
:param str featureset_name: The name of the featureset we should use to extract features from OpenSearch with
:param str ltr_store_name: The name of the LTR store we are using to extract features from
:param int size: The number of results to return in the search result set
:param string terms_field: The name of the field to filter our doc_ids on 
'''
def create_feature_log_query(query, doc_ids, click_prior_query, featureset_name, ltr_store_name, size=200, terms_field="_id"):
    # print("IMPLEMENT ME: create_feature_log_query with proper LTR syntax")
    return {
        'size': size,
        'query': {
            'bool': {
                "filter": [  # use a filter so that we don't actually score anything
                    {
                        "terms": {
                            terms_field: doc_ids
                        }
                    },
                    {  # use the LTR query bring in the LTR feature set
                        "sltr": {
                            "_name": "logged_featureset",
                            "featureset": featureset_name,
                            "store": ltr_store_name,
                            "params": {
                                "keywords": query
                            }
                        }
                    }
                ]
            }
        },
        "ext": {
            "ltr_log": {
                "log_specs": {
                    "name": "log_entry",
                    "named_query": "logged_featureset"
                }
            }
        }
    }


##### Step 4.e:
'''
Modify the query_obj to add a `rescore` entry that uses the baseline query, the LTR information and the rescore window to 
create a new query that actually does the rescoring using our LTR model.
Called from ltr_utils.py

:param str user_query: The user query to run
:param dictionary query_obj: The query object to be submitted to OpenSearch to execute LTR rescoring. Modify this object to add the `rescore` entry with your rescoring query.
:param str click_prior_query: A query string representing what documents were clicked on in the past.  Not used until the 'Using Prior Query History' section of the project.
:param str featureset_name: The name of the featureset we should use to extract features from OpenSearch with
:param str ltr_store_name: The name of the LTR store we are using to extract features from
:param int rescore_size: The number of results to rescore
:param float main_query_weight: A float indicating how much weight to give results that match in the original query
:param float rewcore_query_weight: A float indicating how much weight to give results that match in the rescored query
'''
def create_rescore_ltr_query(user_query: str, query_obj, click_prior_query: str, ltr_model_name: str,
                             ltr_store_name: str,
                             rescore_size=500, main_query_weight=1, rescore_query_weight=2):
    # print("IMPLEMENT ME: create_rescore_ltr_query")
    # query_obj["rescore"] = { ... }

    query_obj["rescore"] = {
        "window_size": rescore_size,
        "query": {
            "rescore_query": {
                "sltr": {
                    "params": {
                        "keywords": user_query,
                        "skus": user_query.split(),
                        "click_prior_query": click_prior_query
                    },
                    "model": ltr_model_name,
                    "store": ltr_store_name
                }
            },
            "score_mode": "total",
            "query_weight": main_query_weight,
            "rescore_query_weight": rescore_query_weight # Magic number, but let's say LTR matches are 2x baseline matches
        },
    }


##### Step Extract LTR Logged Features:
'''
Using the hits object (e.g. response['hits']['hits']) returned by OpenSearch, iterate through the results
and extract the features into a data frame.

:param array hits: The array of hits returned by executing the feature_log_query object against an OpenSearch instance
:param int query_id: The id of the current query we are processing.
'''
def extract_logged_features(hits, query_id):
    import numpy as np
    import pandas as pd
    # print("IMPLEMENT ME: __log_ltr_query_features: Extract log features out of the LTR:EXT response and place in a data frame")
    feature_results = {}
    feature_results["doc_id"] = []  # capture the doc id so we can join later
    feature_results["query_id"] = []  # ^^^
    feature_results["sku"] = []
    feature_results["name_match"] = []
    feature_results["name_match_phrase"] = []
    feature_results["customer_review_average"] = []
    feature_results["customer_review_count"] = []
    feature_results["artist_name_match"] = []
    feature_results["short_description_match"] = []
    feature_results["long_description_match"] = []
    feature_results["sales_rank_short_term"] = []
    feature_results["sales_rank_medium_term"] = []
    feature_results["sales_rank_long_term"] = []
    # rng = np.random.default_rng(12345)
    for (idx, hit) in enumerate(hits):
        feature_results["doc_id"].append(int(hit['_id']))  # capture the doc id so we can join later
        feature_results["query_id"].append(query_id)  # super redundant, but it will make it easier to join later
        feature_results["sku"].append(int(hit['_id']))
        feature_value = hit['fields']['_ltrlog'][0]['log_entry']
        # print(f"FEATURES: {feature_value}")
        feature_results["name_match"].append(feature_value[0]['value'] if 'value' in feature_value[0] else 0) 
        feature_results["name_match_phrase"].append(feature_value[1]['value'] if 'value' in feature_value[1] else 0) 
        feature_results["customer_review_average"].append(feature_value[2]['value'] if 'value' in feature_value[2] else 0) 
        feature_results["customer_review_count"].append(feature_value[3]['value'] if 'value' in feature_value[3] else 0) 
        feature_results["artist_name_match"].append(feature_value[4]['value'] if 'value' in feature_value[4] else 0) 
        feature_results["short_description_match"].append(feature_value[5]['value'] if 'value' in feature_value[5] else 0) 
        feature_results["long_description_match"].append(feature_value[6]['value'] if 'value' in feature_value[6] else 0) 
        feature_results["sales_rank_short_term"].append(feature_value[7]['value'] if 'value' in feature_value[7] else 0) 
        feature_results["sales_rank_medium_term"].append(feature_value[8]['value'] if 'value' in feature_value[8] else 0) 
        feature_results["sales_rank_long_term"].append(feature_value[9]['value'] if 'value' in feature_value[9] else 0) 
    frame = pd.DataFrame(feature_results)
    return frame.astype({'doc_id': 'int64', 'query_id': 'int64', 'sku': 'int64'})