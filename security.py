# security.py
import hashlib
import os
from cryptography.fernet import Fernet
import pandas as pd

# ---------- ANONYMIZATION ----------
def anonymize_df(df, keep_map=False):
    """
    Replace all users in the DataFrame with anonymized IDs.
    Returns anonymized DataFrame. Optionally returns the mapping.
    """
    if 'user' not in df.columns:
        raise ValueError("DataFrame must have a 'user' column for anonymization.")

    mapping = {u: f"User_{hashlib.sha256(u.encode()).hexdigest()[:8]}" for u in df['user'].unique()}
    df2 = df.copy()
    df2['user'] = df2['user'].map(mapping)

    if keep_map:
        return df2, mapping
    return df2


# ---------- ENCRYPTION ----------
def generate_key(key_file='secret.key'):
    """
    Generate a new Fernet key and save to file if it doesn't exist.
    """
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
    return key


def encrypt_bytes(data_bytes, key=None):
    """
    Encrypt bytes using Fernet.
    """
    if key is None:
        key = generate_key()
    f = Fernet(key)
    encrypted = f.encrypt(data_bytes)
    return encrypted


def decrypt_bytes(encrypted_bytes, key=None):
    """
    Decrypt bytes using Fernet.
    """
    if key is None:
        key = generate_key()
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_bytes)
    return decrypted


# ---------- ENCRYPT/DECRYPT FILE ----------
def encrypt_file(input_path, output_path, key=None):
    """
    Encrypt a file from disk and save to output_path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")
    with open(input_path, 'rb') as f:
        data_bytes = f.read()
    encrypted = encrypt_bytes(data_bytes, key)
    with open(output_path, 'wb') as f:
        f.write(encrypted)


def decrypt_file(input_path, output_path, key=None):
    """
    Decrypt an encrypted file from disk.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")
    with open(input_path, 'rb') as f:
        encrypted_bytes = f.read()
    decrypted = decrypt_bytes(encrypted_bytes, key)
    with open(output_path, 'wb') as f:
        f.write(decrypted)
