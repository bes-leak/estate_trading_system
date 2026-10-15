package estate_filters

import "fmt"

type PriceFilter struct {
	Min, Max int64
}

func (c PriceFilter) SQL(args *Args) string {
	return fmt.Sprintf("l.price BETWEEN %s AND %s", args.Add(c.Min), args.Add(c.Max))
}
