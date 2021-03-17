import random
import numpy as np

from milvus import Milvus, IndexType, MetricType, Status

# default to connenct uri address
uri = 'tcp://150.158.219.39:29530'

# vector dimention
_DIM = 8
_INDEX_FILE_SIZE = 10
collection_name = "example_collection_ssc"
partition_tag = "small"

def main():
    milvus = Milvus(uri=uri)
    param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': 32,
            #'metric_type': MetricType.IP
            'metric_type': MetricType.L2
            }
    # show collections in Milvus server
    _, collections = milvus.list_collections()
    
    # 创建 collection
    milvus.create_collection(param)
    # 创建 collection partion
    milvus.create_partition(collection_name, partition_tag)
    
    print(f'collections in Milvus: {collections}')
    # Describe demo_collection
    _, collection = milvus.get_collection_info(collection_name)
    print(f'descript demo_collection: {collection}')
    
    # build fake vectors
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10)]
    vectors1 = [[random.random() for _ in range(_DIM)] for _ in range(10)]
    
    status, id = milvus.insert(
            collection_name=collection_name,
            records=vectors,
            ids=list(range(10)),
            partition_tag=partition_tag)
    print(f'status: {status} | id: {id}')
    if not status.OK():
        print(f"insert failded: {status}")
    
    status1, id1 = milvus.insert(
            collection_name=collection_name,
            records=vectors1,
            ids=list(range(10, 20)),
            partition_tag=partition_tag)
    print(f'status1: {status1} | id1: {id1}')

    ids_deleted = list(range(10))
    
    status_delete = milvus.delete_entity_by_id(collection_name=collection_name,
            id_array=ids_deleted)
    if status_delete.OK():
        print(f'delete successful')

    status, result = milvus.count_entities(collection_name)
    print(f"demo_collection row count: {result}")

    # Flush collection insered data to disk
    milvus.flush([collection_name])
    # Get demo_collection row count
    status, result = milvus.count_entities(collection_name)
    print(f"demo_collection row count: {result}")
    
    # Obtain raw vectors by providing vector ids
    status, result_vectors = milvus.get_entity_by_id(collection_name, list(range(10, 20)))

    # create index of vectors, search more repidly
    index_param = {
            'nlist': 2
            }
    
    # create ivflat index in demo_collection
    status = milvus.create_index(
            collection_name,
            IndexType.IVF_FLAT,
            index_param)
    if status.OK():
        print(f"create index ivf_flat succeeed")

    # use the top 10 vectors for similarity search
    query_vectors = vectors1[0:2]
    
    # execute vector similariy search
    search_param = {
            "nprobe": 16
        }
    
    param = {
            'collection_name': collection_name,
            'query_records': query_vectors,
            'top_k': 3,
            'params': search_param
    }
    
    status, results = milvus.search(**param)
    if status.OK():
        if results[0][0].distance == 0.0:
            print('query result is correct')
        else:
            print('not correct')
        print(results)
    else:
        print(f'search failed: {status}')
    print(results[0][0].distance)

    # 清除已经存在的collection
    milvus.drop_collection(collection_name=collection_name)

    milvus.close()


if __name__ == "__main__":
    main()
