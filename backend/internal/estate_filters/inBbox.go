package estate_filters

import "fmt"

type InBbox struct {
	MinLat, MinLon, MaxLon, MaxLat float64
}

func (c InBbox) SQL(args *Args) string {
	return fmt.Sprintf("ST_Intersects(b.geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))",
		args.Add(c.MinLon), args.Add(c.MinLat), args.Add(c.MaxLon), args.Add(c.MaxLat))
}
