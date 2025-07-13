from astropy.utils.data import download_file
import json

slope=1.1153992589082278
intercept=3.660478564840938
rvalue=0.992890519073263
pvalue=1.4465446260906177e-25
stderr=0.026224244245558064
intercept_stderr=0.3038669495205813

def get_bayestar_snr(superevent):
    root_url = superevent['links']['files']
    fdir = download_file(root_url, cache=True)
    flist = json.load(open(fdir, 'r'))

    try:
        file = download_file(flist['bayestar.html'], cache=True, show_progress=True)
    except:
        return {'bsn':None, 'snr':None, 'snr_err':None}

    with open(file, 'r') as f:
        s = f.read()
        x = s.find('LOGBSN')
        r = s[x:].split('>')[2].split('<')[0]
        bsn = float(r)

        # Fit to 4-ogc based snr
        snr = intercept + slope * (bsn * 2 ) ** 0.5
        snr_err = intercept_stderr + stderr * snr
        return {'bsn':bsn, 'snr':snr, 'snr_err':snr_err}
