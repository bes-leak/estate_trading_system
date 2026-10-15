package estate_filters

import "fmt"

type Criterion interface {
	SQL(args *Args) string
}

type Args struct{ values []any }

func (a *Args) Add(value any) string {
	a.values = append(a.values, value)
	return fmt.Sprintf("$%d", len(a.values))
}

func (a *Args) Values() []any {
	return a.values
}
