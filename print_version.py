import sys
from krakencoder.utils_notorch import get_version

if __name__ == '__main__':
    do_json=len(sys.argv)>1 and sys.argv[1]=='--json'
    version_str=get_version(include_date=True)
    if do_json:
        import json
        print(json.dumps({"krakencoder_version":version_str}))
    else:
        print(f'Krakencoder v{version_str}')