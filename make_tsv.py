from iohandler.datareader import find_files
import re
import pdb

# pdb.set_trace()
fs = find_files('sprite_imgs', '.*\.jpg')
fs = sorted(fs)

# for iWord in range(iFirst, iLast+1):
with open('char10000.tsv', 'w', encoding='utf-8') as f:
	for filename in fs:
		# print 'Word %d: %s' % (iWord, unichr(iWord))
		m = re.search('.*U(\d+)\.jpg', filename)
		# pdb.set_trace()
		char = chr(int(m.group(1)))
		# f.write('{:s}\n'.format(char))
		f.write(char)
		f.write('\n')

