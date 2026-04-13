import numpy as np

def generate_dataset(rows=5, columns=3, rating_range=(1, 5)):
    
    X = np.random.rand(rows, columns) * 10  
    y = np.random.randint(rating_range[0], rating_range[1] + 1, size=rows)

    return X, y

if __name__ == "__main__":
    X, y = generate_dataset()

    np.savetxt("features.txt", X, fmt="%.2f", header="Feature Matrix", comments="")
    np.savetxt("ratings.txt", y, fmt="%d", header="Ratings Vector", comments="")

    print("Success !")