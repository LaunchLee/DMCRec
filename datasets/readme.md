# Datasets

All dataset resources are shared via Google Drive:

**Google Drive Folder:**  
<https://drive.google.com/drive/folders/1jnsSNG8FH8zkYaGXiZ2pxzey2eeC_k5l?usp=sharing>

---

## 📁 Cache Files

Files inside the **`Cache`** directory on Google Drive should be placed into:

```
cache/_embeddings
```

---

## 📁 Cornac-Related Files

The **`Cornac`** directory contains:

- caches downloaded automatically by **`cornac`**
- caches downloaded by **`transformers`**
- the joblib embedding file

These embedding files should also be placed into:

```
cache/_embeddings
```

> **Note:**  
> Since these caches can be downloaded automatically by the code—and the *Cornac clothing dataset* is small enough that generating embeddings is fast—downloading them manually is optional.

---

## 📦 Dataset Archives

All dataset `.zip` files in Google Drive should be extracted **directly into**:

```
datasets
```
