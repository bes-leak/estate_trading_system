package estate_filters

import "fmt"

// RoomsIn: слайс уходит одним параметром, pgx сам превратит его в массив postgres
type RoomsIn struct {
	Rooms []int16
}

func (c RoomsIn) SQL(args *Args) string {
	return fmt.Sprintf("l.rooms = ANY(%s)", args.Add(c.Rooms))
}

type AreaRange struct {
	Min, Max float32
}

func (c AreaRange) SQL(args *Args) string {
	return fmt.Sprintf("l.area BETWEEN %s AND %s", args.Add(c.Min), args.Add(c.Max))
}

// NotFirstFloor - лист вообще без параметров
type NotFirstFloor struct{}

func (c NotFirstFloor) SQL(args *Args) string {
	return "l.level > 1"
}

// MetroWithin - условие по признаку дома, поэтому префикс b.
type MetroWithin struct {
	Meters float64
}

func (c MetroWithin) SQL(args *Args) string {
	return fmt.Sprintf("b.distance_metro <= %s", args.Add(c.Meters))
}
