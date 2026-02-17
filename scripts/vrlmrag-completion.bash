#!/bin/bash
# vrlmrag shell completion script
# Source this file or add to your shell config:
#   source /path/to/vrlmrag-completion.sh

_vrlmrag_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main options
    opts="--query --output --format --provider --model --profile --verbose --quiet
          --text-only --no-embed --cache --dry-run --local --offline
          --max-depth --max-iterations --interactive --store-dir
          --chunk-size --chunk-overlap --use-sqlite
          --rrf-dense-weight --rrf-keyword-weight --multi-query
          --reindex --rebuild-kg --model-compare --check-model
          --quality-check --export-graph --graph-format --graph-stats
          --deduplicate-kg --dedup-threshold --dedup-report
          --graph-augmented --graph-hops
          --collection --add --collection-list --collection-info
          --collection-delete --collection-export --collection-import
          --import-rename --collection-merge --collection-description
          --collection-tag --collection-untag --collection-search
          --collection-search-tags --collection-stats --global-stats
          --list-providers --show-hierarchy --lock-status"
    
    case "${prev}" in
        --provider|-p)
            COMPREPLY=( $(compgen -W "openrouter zenmux zai openai anthropic gemini litellm sambanova nebius modalresearch ollama" -- "${cur}") )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "markdown json" -- "${cur}") )
            return 0
            ;;
        --profile)
            COMPREPLY=( $(compgen -W "fast thorough comprehensive" -- "${cur}") )
            return 0
            ;;
        --graph-format)
            COMPREPLY=( $(compgen -W "mermaid graphviz networkx" -- "${cur}") )
            return 0
            ;;
        --collection|-c)
            # Complete with available collection names
            local collections
            collections=$(vrlmrag --collection-list 2>/dev/null | grep "^  -" | sed 's/^  - //' || echo "")
            COMPREPLY=( $(compgen -W "${collections}" -- "${cur}") )
            return 0
            ;;
        --output|-o|--export-graph|--collection-export|--collection-import)
            # File path completion
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --add|--input)
            # File/directory path completion
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
    esac
    
    # Complete with options or files
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    else
        COMPREPLY=( $(compgen -f -- "${cur}") )
    fi
}

complete -F _vrlmrag_complete vrlmrag
