from pamap import PAMAP

pamap = PAMAP( PAMAP.PROCESSED )

data = pamap.cross_validation( 1 )
print data['train']
print data['test']
