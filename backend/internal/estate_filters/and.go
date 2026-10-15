package estate_filters

import "strings"

type And struct {
	Children []Criterion
}

func (c And) SQL(args *Args) string {
	if len(c.Children) == 0 {
		return "TRUE"
	}

	parts := make([]string, 0, len(c.Children))

	for _, child := range c.Children {
		parts = append(parts, child.SQL(args))
	}

	return "(" + strings.Join(parts, " AND ") + ")"
}
