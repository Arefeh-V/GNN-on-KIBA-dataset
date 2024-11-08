from gensim.models import Word2Vec
import numpy as np

def generate_deepwalk_embeddings(graph, dimensions, num_walks, walk_length):
    # Generate random walks
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    
    # Train Word2Vec on walks (DeepWalk approach)
    model = Word2Vec(walks, vector_size=dimensions, window=10, min_count=1, sg=1, workers=4)
    model.build_vocab(walks) 
    total_words = sum(len(walk) for walk in walks)

    model.train(walks, total_words=total_words, epochs=model.epochs)
    
    return model
