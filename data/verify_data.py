import hashlib
import os

CHECKSUM_MAP = {
    'annotations/labeled.json': '49f674dd3d738fc46f66a6548dce7ba5',
    'annotations/test_a.json': '8888045bd6b67533599cadc05a52a3b4',
    'annotations/unlabeled.json': '6ffb2fa1f729071aff52651b24f90f39',
    'zip_feats/labeled.zip': '4609a40b0df91e51bc713527de920b00',
    'zip_feats/test_a.zip': '531cf10c1ded29799c98324f36d9e90a',
    'zip_feats/unlabeled.zip': '732f649429250f35476a11d01e0b5258',
}


def get_checksum(file_path):
    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        block_size = 2 ** 20
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    checksum = md5.hexdigest()
    return checksum


def verify_data(data_dir):
    """
    to verify the integrity of the whole downloaded data directory
    :param data_dir: the downloaded data directory
    :return: files missing or failed in the verification
    """
    missing = []
    failed = []
    for file, checksum in CHECKSUM_MAP.items():
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            print(f'ERROR: file "{file}" is MISSING in "{data_dir}"')
            missing.append(file)
            continue
        if checksum != get_checksum(file_path):
            print(f'ERROR: checksum is inconsistent for file "{file}"')
            failed.append(file)
        else:
            print(f'checksum is consistent for file "{file}"')
    return missing, failed


if __name__ == '__main__':
    missing, failed = verify_data('./')
    if missing or failed:
        print(f'Verification FAILED, please retry downloading: {missing + failed}')
    else:
        print('Verification Passed!')
