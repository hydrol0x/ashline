#!/bin/sh
# Downloads 14 free Pexels photos of families at home, across 3 rooms, into ~/Desktop/ashline-demo.
# Pexels License: free to use, no attribution required. Credit is still kind:
#   living room and bedroom: Gustavo Fring; kitchen and desk: August de Richelieu (both on Pexels).
set -e
DIR="$HOME/Desktop/ashline-demo"
mkdir -p "$DIR"
cd "$DIR"
get() {  # name, pexels photo id
  curl -sSL -o "$1.jpg" "https://images.pexels.com/photos/$2/pexels-photo-$2.jpeg?auto=compress&cs=tinysrgb&w=1600"
  echo "  $1.jpg"
}
echo "Living room (the same sofa from several angles):"
get 01-living-room-tea 3875141
get 02-living-room-sofa 3875142
get 03-living-room-family 3875157
get 04-living-room-mugs 3875160
get 05-living-room-kids 3875175
echo "Bedroom:"
get 06-bedroom-flowers 3875143
get 07-bedroom-dad-and-baby 3875150
get 08-bedroom-weekend 3875173
get 09-bedroom-reading 3875332
echo "Kitchen:"
get 10-kitchen-cooking 4261782
get 11-kitchen-dinner 4261791
get 12-kitchen-pasta 4261797
get 13-kitchen-family 4261798
echo "Study desk:"
get 14-desk-laptop 4261785
echo "Done: $DIR"
open "$DIR" 2>/dev/null || true
