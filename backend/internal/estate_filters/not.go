package estate_filters

type Not struct {
	Child Criterion
}

func (c Not) SQL(args *Args) string {
	if c.Child == nil {
		return "TRUE"
	}

	return "NOT (" + c.Child.SQL(args) + ")"
}
